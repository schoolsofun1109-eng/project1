import os
import re
import shlex
import subprocess
import torch

from PyTorchSimFrontend import extension_config
from torch._inductor.codecache import get_hash, write
from torch._inductor.async_compile import AsyncCompile
from AsmParser.tog_generator import tog_generator
from PyTorchSimFrontend.mlir.mlir_caller_codegen import MLIRKernelCallerCodeGen
from Simulator.simulator import FunctionalSimulator, CycleSimulator, TOGSimulator

# Configure logger for extension_codecache module (WARNING level by default)
logger = extension_config.setup_logger()

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

def _rect_height_opt(vectorlane_size):
    """Extra ``systolic-array-k-depth=<Sm*Sk>`` token for the pytorchsim-to-vcix pass.

    Sm*Sk is the array's K capacity, not a fudge factor. This array is
    weight-stationary, so its stationary tile is K x N and its two physical
    axes are height = K and width = N; M is streamed through over time and is
    not a physical axis at all. pe_M/systolic_array_height is therefore the K
    axis despite its name, and Sk (pe_P) adds a per-PE adder tree that reduces
    Sk more K values per cycle. Both feed K, so one array pass consumes
    height * Sk = Sm * Sk values of K -- which is exactly the granularity K
    must be tiled by.

    Do not "correct" this to Sk alone. That was tried; the only functional-mode
    config it appeared to validate against (systolic_ws_rect_sk2_func:
    Sm=64,Sk=2) has Sm*Sk == Sn == 128, which trips the early return below and
    emits no flag at all -- i.e. that test never exercised a 3D path in the
    first place and proves nothing either way. See extension_config.py for the
    gem5 / Spike / MLIR evidence that height is the K axis.
    """
    if not extension_config.systolic_array_real_rect:
        return ""
    sm = extension_config.systolic_array_height or vectorlane_size
    sk = extension_config.systolic_array_size_k or 1

    if extension_config.systolic_dataflow == "os":
        # Output-stationary 3D: the array face is Sm x Sn and each PE owns one
        # C[m][n] accumulator, so M is tiled by Sm and K by Sk alone (Sk is the
        # per-PE adder-tree depth). See rect_array_work/OS_3D_CONTRACT.md.
        return (f" systolic-dataflow=os"
                f" systolic-array-m-height={sm}"
                f" systolic-array-k-depth={sk}")

    # Weight-stationary (legacy): the array face is K x N, so the height IS the
    # K axis and one array pass consumes height * Sk values of K. That product,
    # not Sk, is the K-tiling granularity.
    k_per_pass = sm * sk
    if k_per_pass == vectorlane_size:
        return ""
    return f" systolic-array-k-depth={k_per_pass}"

def mlir_compile_command(filename, vectorlane_size, vlen=256):
    return [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding \
            -dma-fine-grained='systolic-array-size={vectorlane_size}' \
            -global-idx='vlen={vlen}' \
            -test-pytorchsim-to-vcix='systolic-array-size={vectorlane_size}{_rect_height_opt(vectorlane_size)} vlen={vlen}' \
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

            # Create TOG
            w_offset, x_offset = vectorlane_size, vectorlane_size
            if kwargs['loop_size'] is not None and kwargs['loop_size'][-3] < vectorlane_size:
                x_offset = kwargs['loop_size'][-3]
            if kwargs['loop_size'] is not None and kwargs['loop_size'][-1] < vectorlane_size:
                w_offset = kwargs['loop_size'][-1]
            w_offset = 0 # max(w_offset - x_offset, 0)
            tile_graph_generator = tog_generator(origins)
            tile_graph_generator.load_file(raw_tog_path)
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
