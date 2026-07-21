import os
import re
import math
import shlex
import subprocess
import torch

from PyTorchSimFrontend import extension_config
from torch._inductor.codecache import get_hash, write
from torch._inductor.async_compile import AsyncCompile
from AsmParser.tog_generator import tog_generator
from AsmParser.onnx_utility import compute_node
from PyTorchSimFrontend.mlir.mlir_caller_codegen import MLIRKernelCallerCodeGen
from Simulator.simulator import FunctionalSimulator, CycleSimulator, TOGSimulator

# Configure logger for extension_codecache module (WARNING level by default)
logger = extension_config.setup_logger()

def _scale_cycles_for_3d_pe(cycle_list, compute_types, k_tile, plane_size):
    """Post-correct gem5-measured cycles for a rectangular/3D (Sm x Sn x Sk) PE array.

    gem5 always measures a SQUARE array of side Sn = plane_size (= vpu_num_lanes),
    with the MLIR tiling matched to that same square. For that square measurement
    the K reduction takes ceil(K/Sn) passes. A target array of height Sm and depth
    Sk reduces Sm*Sk of K per pass -> ceil(K/(Sm*Sk)) passes. Since the per-tile
    cycle is K-independent (M-streaming + fill/drain; verified empirically) and the
    WIDTH already equals Sn (so N-tiling and width fill/drain are measured exactly),
    only the K-pass count needs correcting:

        ratio = ceil(K/(Sm*Sk)) / ceil(K/Sn)
        c_3d  = round(c * ratio)

    Cases:
      * Sm == Sn and Sk == 1 (square 2D): ratio = 1 -> bit-exact, no change.
      * Sm == Sn, Sk > 1 (n x n x k depth): ratio = ceil(K/(Sn*Sk))/ceil(K/Sn).
      * Sm != Sn (rectangular height): ratio = ceil(K/(Sm*Sk))/ceil(K/Sn).
      * K <= Sm*Sk (single pass): ratio = 1 -> no speedup (nothing to parallelize).

    Only GEMM/preload nodes (compute_type > 0) are corrected; vector nodes
    (compute_type == 0, e.g. softmax / RMSNorm / RoPE) have no K reduction.

    Note (approximation): the K-direction fill/drain uses the measured Sn rather
    than the target Sm, and M-tiling is left at the measured square. These are
    second-order vs. the K-pass count. Faithful rectangular tiling would need the
    (prebuilt, source-unavailable) MLIR pass + gem5 rebuild.
    """
    Sk = extension_config.systolic_array_size_k or 1
    Sn = plane_size                                        # gem5 array width
    Sm = extension_config.systolic_array_height or Sn      # array fill height (one layer)
    real = extension_config.systolic_array_real_rect

    # Real mode: gem5 + the MLIR passes model the array directly and with no
    # post-scale. The 3D depth Sk is handled by DECOUPLING the two knobs --
    # the MLIR K-tiling divisor is Sm*Sk (K reduced per pass) while gem5's fill
    # height stays Sm (--vlane-height) so saSize = Sm+Sn-1. Both the pass count
    # and the fill are therefore really simulated, so return the cycles as-is.
    if real:
        return cycle_list

    # ---- Legacy post-scale approximation (real mode off) ----
    # gem5 measured a square Sn x Sn array (K tiled by Sn). The target array of
    # height Sm and depth Sk reduces (Sm*Sk) of K per pass; rescale by the
    # pass-count ratio. Kept for backward compatibility with old configs.
    measured_height = Sn
    if Sk <= 1 and Sm == Sn:
        return cycle_list                          # square 2D: identical to before

    if len(cycle_list) != len(compute_types):
        # Should not happen (one cycle per compute node); stay safe rather than corrupt.
        logger.warning("[3D-PE] cycle/compute-node count mismatch (%d vs %d); skipping correction",
                       len(cycle_list), len(compute_types))
        return cycle_list
    if k_tile and k_tile > 0:
        passes_measured = math.ceil(k_tile / measured_height)
        passes_target = math.ceil(k_tile / (Sm * Sk))      # 3D: Sm height x Sk depth
        ratio = passes_target / passes_measured
    else:
        ratio = None
    scaled = []
    for c, ctype in zip(cycle_list, compute_types):
        if ctype is None or ctype <= 0:
            scaled.append(c)                        # vector node: no K reduction, keep as-is
        elif ratio is None:
            scaled.append(max(1, math.ceil(c / Sk)))  # K tile unknown: conservative
        else:
            scaled.append(max(1, round(c * ratio)))   # GEMM: corrected K-pass count
    logger.info("[3D-PE] real=%s Sm=%d Sn=%d Sk=%d K_t=%s base_h=%d pass_ratio=%s: cycles %s -> %s (GEMM only)",
                real, Sm, Sn, Sk, k_tile, measured_height,
                None if ratio is None else round(ratio, 3), cycle_list, scaled)
    return scaled

def _weight_feed_is_concurrent():
    """True when the weight feed should overlap the input feed instead of
    running ahead of it.

    gem5 times vwpush and vipush as two separate RISC-V instructions and TOGSim
    then chains the two nodes, so a pass costs W_feed + A_feed. But inside the
    modelled array the two feeds are independent (separate wQueue/iQueue, and
    the SystolicArray debug trace shows osCompute starting on the very cycle the
    A-slice lands, with no wait on the weight), so charging them back to back
    double-counts. Only WS genuinely serializes: its stationary tile must be
    fully resident before the M stream starts.
    """
    return (extension_config.operand_delivery == "concurrent"
            and extension_config.systolic_dataflow == "os")


LOCK_TIMEOUT = 600

def hash_prefix(hash_value):
    return hash_value[1:12]

def get_write_path(src_code):
    return os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, hash_prefix(get_hash(src_code.strip())))


def get_lock_path(write_path):
    """Return lock file path for the given write_path (per-source_code lock)."""
    return os.path.join(write_path, ".compile.lock")

def dump_metadata(args, arg_attributes, path):
    meta_path = os.path.join(path, "meta.txt")
    if os.path.isfile(meta_path):
        return

    with open(meta_path, "a") as file:
        for (arg_name, arg_attribute), arg in zip(arg_attributes, args):
            file.write(f'{arg_name}=({arg_attribute[0]}, {arg.dtype}, {arg.shape})\n')
    return

def _rect_height_opt(vectorlane_size, force_ws=False):
    """Extra ``systolic-array-height=<Sm*Sk>`` token for the pytorchsim-to-vcix pass.

    The depth Sk's dominant timing effect is the reduced PASS COUNT: a 3D array
    of height Sm and depth Sk reduces Sm*Sk of K per pass, so K is tiled by
    Sm*Sk (passes = K/(Sm*Sk)). This is expressed through the MLIR instruction
    stream (fewer vpush/compute), which is what actually drives gem5's cycles.
    gem5 additionally receives Sm (--vlane-height, fill) and Sk (--vlane-depth,
    adder-tree latency) so it reports the true 3D geometry (WxHxD). Empty when
    Sm*Sk == Sn (== the default square tiling) or real mode is off.

    ``force_ws`` compiles the WS kernel even when dataflow==os. Used for the
    FUNCTIONAL (Spike / accuracy) binary: OS and WS produce identical matmul
    VALUES, so the value-check runs the validated WS kernel while only the
    TIMING (gem5) binary uses the OS kernel/timing. This keeps values correct
    regardless of the OS Spike accumulator path.
    """
    if not extension_config.systolic_array_real_rect:
        return ""
    sm = extension_config.systolic_array_height or vectorlane_size
    sk = extension_config.systolic_array_size_k or 1
    if extension_config.systolic_dataflow == "os":
        if force_ws:
            # Functional accuracy binary for an OS config: matmul VALUES are
            # dataflow-independent, so emit the plain square WS kernel (Sm/Sk are
            # OS-only geometry). A square Sn x Sn WS matmul yields identical
            # values and stays on the validated Spike WS path.
            return ""
        # Output-stationary 3D: the array face is Sm x Sn and each PE owns one
        # C[m][n] accumulator, so M is tiled by Sm and K by Sk alone (Sk is the
        # per-PE adder-tree depth). The MLIR pass emits the N->M->K OS kernel.
        return (f" systolic-dataflow=os"
                f" systolic-array-m-height={sm}"
                f" systolic-array-k-depth={sk}")
    # Weight-stationary (legacy): the array face is K x N, so height IS the K
    # axis and one array pass consumes height * Sk values of K.
    k_per_pass = sm * sk
    if k_per_pass == vectorlane_size:
        return ""
    return f" systolic-array-height={k_per_pass}"

def mlir_compile_command(filename, vectorlane_size, vlen=256):
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding \
            -dma-fine-grained='systolic-array-size={vectorlane_size}' \
            -global-idx='vlen={vlen}' \
            -test-pytorchsim-to-vcix='systolic-array-size={vectorlane_size}{_rect_height_opt(vectorlane_size, force_ws=True)} vlen={vlen}' \
            -test-memref-to-gemmini="vectorlane={vectorlane_size}" \
            -convert-linalg-to-loops \
            -convert-vector-to-scf='full-unroll' \
            -lower-affine \
            -finalize-memref-to-llvm \
            -lower-vector-multi-reduction \
            -convert-vector-to-llvm \
            -convert-arith-to-llvm \
            -convert-math-to-llvm \
            -convert-scf-to-cf \
            -convert-cf-to-llvm \
            -convert-func-to-llvm \
            -convert-index-to-llvm \
            -reconcile-unrealized-casts \
            {'--mlir-print-ir-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_MLIR_IR else ''} \
            {filename}.mlir -o {filename}_llvm.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {filename}_llvm.mlir -o {filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -filetype=obj \
                {'--print-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_LLVM_IR else ''} \
                -O2 {filename}.ll -o {filename}.o
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -O2 {filename}.ll -o {filename}.s
        """,
    ).strip()]

def mlir_gem5_compile_command(filename, sample_filename, tog_file, vectorlane_size, vlen=256):
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding='timing_mode=1' \
            -dma-fine-grained='systolic-array-size={vectorlane_size}' \
            -global-idx='vlen={vlen}' \
            -test-pytorchsim-to-vcix='systolic-array-size={vectorlane_size}{_rect_height_opt(vectorlane_size)} vlen={vlen}' \
            -test-tile-operation-graph='vectorlane={vectorlane_size} sample-mode={extension_config.CONFIG_TLS_MODE}' \
            -test-memref-to-gemmini="vectorlane={vectorlane_size} timing=1" \
            -convert-linalg-to-loops \
            -convert-vector-to-scf='full-unroll' \
            -lower-affine \
            -finalize-memref-to-llvm \
            -lower-vector-multi-reduction \
            -convert-vector-to-llvm \
            -convert-arith-to-llvm \
            -convert-math-to-llvm \
            -convert-scf-to-cf \
            -convert-cf-to-llvm \
            -convert-func-to-llvm \
            -convert-index-to-llvm \
            -reconcile-unrealized-casts \
            {'--mlir-print-ir-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_MLIR_IR else ''} \
            {filename}.mlir -o {sample_filename}_llvm.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {sample_filename}_llvm.mlir -o {sample_filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -filetype=obj \
                {'--print-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_LLVM_IR else ''} \
                -O2 {sample_filename}.ll -o {sample_filename}.o
        """,
    ).strip()]

class SpadOverflowError(Exception):
    def __init__(self, message="SPAD overflow occurred."):
        super().__init__(message)

class TileSizeError(Exception):
    def __init__(self, message="SPAD overflow occurred."):
        super().__init__(message)

class MLIRCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)   # Todo: Cache

    @staticmethod
    def _load_library(path):
        pass

    @classmethod
    def load(cls, source_code,
             validation_wrapper_name="validation_wrapper",
             validation_binary_name="validation_bin",
             cycle_wrapper_name="cycle_wrapper",
             cycle_binary_name="cycle_bin",
             arg_attributes=[], vectorlane_size=16,
             spad_info=None, origins=None, silent_mode=False, **kwargs):
        vlen = kwargs['vlen']
        vlenb = vlen // 8
        write_path = get_write_path(source_code)
        key, input_path = write(source_code, "mlir", specified_dir=write_path)
        new_input_path = os.path.splitext(input_path)[0]
        raw_tog_path = new_input_path + "_tog.py"
        tog_path = os.path.join(write_path, "tile_graph.onnx")
        sample_mlir_path = new_input_path + "_sample"
        validation_binary_path = os.path.join(write_path, validation_binary_name)
        gem5_cmds = mlir_gem5_compile_command(new_input_path, sample_mlir_path, raw_tog_path, vectorlane_size)

        from filelock import FileLock
        os.makedirs(write_path, exist_ok=True)
        lock = FileLock(get_lock_path(write_path), timeout=LOCK_TIMEOUT)

        if spad_info is not None and 'imem_vaddr' in spad_info:
            # 3-way IMEM/WMEM/OMEM split: place each bank's linker section at
            # its own base address so weight/output tiles actually land in
            # their dedicated region instead of being packed into IMEM.
            link_option = (
                f"-Wl,--section-start=.imem=0x{spad_info['imem_vaddr']:x} "
                f"-Wl,--section-start=.wmem=0x{spad_info['wmem_vaddr']:x} "
                f"-Wl,--section-start=.omem=0x{spad_info['omem_vaddr']:x}"
            )
        elif spad_info is not None:
            link_option = f"-Wl,--section-start=.spad=0x{spad_info['spad_vaddr']:x}"
        else:
            link_option = ""
        # Generate LLVM kernel calller and binary for validation
        if extension_config.pytorchsim_functional_mode:
            # Use custom malloc to avoid size error
            new_link_option = link_option + " -Wl,--wrap=malloc -Wl,--wrap=free"
            cmds = mlir_compile_command(new_input_path, vectorlane_size, vlen=vlen)
            opt_cmd = shlex.split(cmds[0])
            translate_cmd = shlex.split(cmds[1])
            llc_cmd = shlex.split(cmds[2])
            llc_asm_cmd = shlex.split(cmds[3])
            with lock:
                try:
                    subprocess.check_call(opt_cmd)
                    subprocess.check_call(translate_cmd)
                    subprocess.check_call(llc_cmd)
                    subprocess.check_call(llc_asm_cmd)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Command failed with exit code {e.returncode}")
                    logger.error(f"Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
                    assert(0)

                val_llvm_caller = MLIRKernelCallerCodeGen(extension_config.pytorchsim_functional_mode, arg_attributes)
                val_llvm_caller.generate_wrapper_file(write_path, validation_wrapper_name)
                val_llvm_caller.compile_wih_kernel(write_path, key, validation_wrapper_name,
                                                   validation_binary_name, new_link_option)

                stack_size = val_llvm_caller.parse_stack_sizes(f"{write_path}/{key}.s", vlenb=vlenb)
                spad_size =  val_llvm_caller.get_spad_size(validation_binary_path)
                spad_usage = stack_size + spad_size # Spad usage per lane
                if extension_config.CONFIG_SPAD_INFO["spad_size"] < spad_usage:
                    logger.debug(
                        f"Scratchpad size exceeded: required {spad_usage} bytes, "
                        f"but only {extension_config.CONFIG_SPAD_INFO['spad_size']} bytes available."
                    )
                    raise SpadOverflowError()

        # Skip if TOG file already exists
        if os.path.isfile(tog_path):
            return key

        # Launch tile graph generator
        gem5_sample_cmd = shlex.split(gem5_cmds[0])
        gem5_translate_cmd = shlex.split(gem5_cmds[1])
        gem5_llc_cmd = shlex.split(gem5_cmds[2])

        lock = FileLock(get_lock_path(write_path), timeout=LOCK_TIMEOUT)
        with lock:
            try:
                result = subprocess.check_output(gem5_sample_cmd)
                with open(raw_tog_path, "wb") as file:
                    file.write(result)
                subprocess.check_call(gem5_translate_cmd)
                subprocess.check_call(gem5_llc_cmd)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                logger.error(f"Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
                assert(0)

            if not extension_config.pytorchsim_timing_mode:
                return key

            # Generate MLIR kernel calller and binary for cycle calculation
            cycle_llvm_caller = MLIRKernelCallerCodeGen(False, arg_attributes, cycle_sim=True)
            cycle_llvm_caller.generate_wrapper_file(write_path, cycle_wrapper_name)
            cycle_llvm_caller.compile_wih_kernel(write_path, key + "_sample", cycle_wrapper_name, cycle_binary_name, link_option)

            # Run cyclesim
            cyclesim = CycleSimulator()
            cycle_list = cyclesim.compile_and_simulate(os.path.join(write_path, cycle_binary_name), vectorlane_size, silent_mode=silent_mode)

            # 3D PE array: K tile length for the depth-Sk scaling. The scaling itself
            # is applied below, AFTER the TOG is loaded, so it can see compute-node
            # types and skip vector nodes. loop_size = [TOG_latency, TILE_N, TILE_K].
            k_tile = kwargs['loop_size'][-1] if kwargs.get('loop_size') else None

            # Create TOG.
            # offset = the cycles a compute node holds the array ENTRANCE, i.e.
            # how long before the next node can start feeding. TOGSim turns it
            # into SA active via overlapping_cycle = compute_cycle - offset.
            if extension_config.systolic_dataflow == "os":
                # Output-stationary: the array face is Sm x Sn and BOTH operands
                # stream every K-step, so the WS proxy "array width" no longer
                # describes what occupies the entrance. What actually does is the
                # push count: one pass issues ceil(Sm/nr_element) vipush and
                # ceil(Sk/nr_element) vwpush (confirmed against the instruction
                # trace), and each push is one transfer through the entrance.
                # Unlike WS -- where one weight tile is amortized over a long M
                # stream -- OS reloads weights every pass, so w_offset is NOT 0.
                _nr = max(1, extension_config.vpu_vector_length_bits // 32)
                _sm = extension_config.systolic_array_height or vectorlane_size
                _sk = extension_config.systolic_array_size_k or 1
                x_offset = max(1, -(-_sm // _nr))
                w_offset = max(1, -(-_sk // _nr))
                if _weight_feed_is_concurrent():
                    # Separate W and A ports: the weight feed rides its own wires,
                    # so it does not consume the input port's bandwidth. offset=0
                    # makes overlapping_cycle = compute_cycle (see tog_generator),
                    # i.e. the feed keeps its full gem5-measured duration in the
                    # trace but adds zero exclusive array time.
                    #
                    # Do NOT shrink compute_cycle itself to fake this. One vwpush
                    # moves n_vu x vl elements; lanes are parallel but the vl
                    # elements within a lane enter that lane's serializer one per
                    # cycle, so the measured ~17 cycles is real transfer work.
                    w_offset = 0
            else:
                # Weight-stationary (unchanged): entrance time is approximated by
                # the array width, and the stationary weight tile is amortized
                # over the M stream, so its load hides behind the input feed.
                w_offset, x_offset = vectorlane_size, vectorlane_size
                if kwargs['loop_size'] is not None and kwargs['loop_size'][-3] < vectorlane_size:
                    x_offset = kwargs['loop_size'][-3]
                if kwargs['loop_size'] is not None and kwargs['loop_size'][-1] < vectorlane_size:
                    w_offset = kwargs['loop_size'][-1]
                w_offset = 0 # max(w_offset - x_offset, 0)
            tile_graph_generator = tog_generator(origins)
            tile_graph_generator.load_file(raw_tog_path)
            # Now that compute-node types are known, apply the 3D (Sk) scaling to GEMM
            # nodes only (compute_type > 0), leaving vector nodes untouched. The node
            # order here matches the cycle_list pop order in generate_tile_graph.
            compute_types = [n.torchsim_compute_type
                             for n in tile_graph_generator.node_dict.values()
                             if isinstance(n, compute_node)]
            cycle_list = _scale_cycles_for_3d_pe(cycle_list, compute_types, k_tile, vectorlane_size)
            tile_graph_generator.generate_tile_graph(
                tog_path,
                cycle_list=cycle_list,
                x_offset=x_offset, # FIXME.
                w_offset=w_offset, # FIXME.
                vector_lane=vectorlane_size
            )
        return key

class CustomAsyncCompile(AsyncCompile):
    def __init__(self):
        self.validation_wrapper_name = "validation_wrapper"
        self.validation_binary_name = "validation_binary"
        self.cycle_wrapper_name = "cycle_wrapper"
        self.cycle_binary_name = "cycle_binary"

    def mlir(self, source_code, arg_attributes=[], vectorlane_size=16, tile_size=[], spad_info=None, origins=None, silent_mode=False, **kwargs):
        autotune = kwargs.get('autotune', False)
        def task():
            key = MLIRCodeCache.load(source_code,
                                          valdiation_wrapper_name=self.validation_binary_name,
                                          validation_binary_name=self.validation_binary_name,
                                          arg_attributes=arg_attributes, vectorlane_size=vectorlane_size,
                                          tile_size=tile_size, spad_info=spad_info, origins=origins,
                                          silent_mode=autotune, **kwargs)
            return key
        future = self.submit(task)

        def run_kernel_simulation(*args, autotune_subprocess_timeout_sec=None, **kwargs):
            # Wait for compilation
            key = future.result()
            from filelock import FileLock
            result_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, hash_prefix(key))
            lock = FileLock(get_lock_path(result_path), timeout=LOCK_TIMEOUT)
            with lock:
                # Run simulator pass
                # Dump arguments and meta data
                dump_metadata(args, arg_attributes, result_path)
                runtime_path = FunctionalSimulator.get_runtime_dump_path(result_path)
                if extension_config.pytorchsim_functional_mode and not autotune:
                    funcsim = FunctionalSimulator(result_path, key)
                    funcsim.run_spike(args, arg_attributes,
                                    runtime_path, self.validation_binary_name,
                                    vectorlane_size=vectorlane_size, spad_info=spad_info,
                                    silent_mode=autotune)

                if not extension_config.pytorchsim_timing_mode:
                    return [float("inf")]

                # Prepare arguments for launch kernel
                onnx_path = os.path.join(result_path, "tile_graph.onnx")
                attribute_dir = os.path.join(runtime_path, "attribute")
                kernel_attribute_path = TOGSimulator.write_kernel_attribute_file(attribute_dir, args)

                TOGSim = torch.npu.get_tog_simulator()
                if not autotune and TOGSim is not None:
                    torch.npu.launch_kernel(onnx_path, kernel_attribute_path)
                    result = None # No result for non-autotune mode
                else:
                    result_path = TOGSimulator.run_standalone(
                        onnx_path,
                        kernel_attribute_path,
                        autotune_mode=autotune,
                        timeout_sec=autotune_subprocess_timeout_sec,
                    )
                    result = TOGSimulator.get_result_from_file(result_path)
                return result
        return run_kernel_simulation
