import os
import sys
import importlib
import yaml
import logging

CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
CONFIG_GEM5_PATH = os.environ.get('GEM5_PATH', default="/workspace/gem5/build/RISCV/gem5.opt")
CONFIG_TORCHSIM_LLVM_PATH = os.environ.get('TORCHSIM_LLVM_PATH', default="/usr/bin")

CONFIG_TORCHSIM_TOG_HOST_CC = os.environ.get("TORCHSIM_TOG_HOST_CC", "gcc")

def _default_tog_host_cflags():
    """Host flags for ``dlopen``'d ``*_tog.so`` / ``tile_operation_graph.so``."""
    if os.environ.get("TORCHSIM_TOG_HOST_CFLAGS"):
        return os.environ["TORCHSIM_TOG_HOST_CFLAGS"]
    if True: #int(os.environ.get("TORCHSIM_TOG_SO_DEBUG", "0")):
        return (
            "-g -Og -fno-omit-frame-pointer -fPIC -std=c11 "
            "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter"
        )
    return (
        "-O2 -fPIC -std=c11 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter"
    )


CONFIG_TORCHSIM_TOG_HOST_CFLAGS = _default_tog_host_cflags()


def _default_tog_host_ldflags():
    if os.environ.get("TORCHSIM_TOG_HOST_LDFLAGS"):
        return os.environ["TORCHSIM_TOG_HOST_LDFLAGS"]
    # Keep debug sections in .so; optional build-id helps GDB locate DWARF.
    base = "-shared"
    if int(os.environ.get("TORCHSIM_TOG_SO_DEBUG", "0")):
        return base + " -Wl,--build-id"
    return base


CONFIG_TORCHSIM_TOG_HOST_LDFLAGS = _default_tog_host_ldflags()

CONFIG_TORCHSIM_DUMP_MLIR_IR = int(os.environ.get("TORCHSIM_DUMP_MLIR_IR", default=False))
CONFIG_TORCHSIM_DUMP_LLVM_IR = int(os.environ.get("TORCHSIM_DUMP_LLVM_IR", default=False))

def __getattr__(name):
    # TOGSim config
    config_path = os.environ.get('TOGSIM_CONFIG',
                default=f"{CONFIG_TORCHSIM_DIR}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml")
    if name == "CONFIG_TOGSIM_CONFIG":
        return config_path

    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)

    # Hardware info config.
    #
    # WHAT THE PE ARRAY ACTUALLY IS (read this before touching pe_M):
    # This is a weight-stationary systolic array. The stationary weight tile is
    # K x N, so the array's two physical axes are HEIGHT = K (the reduction
    # axis) and WIDTH = N (the output-channel axis). M is NOT a physical axis:
    # activations are streamed through the array over time, M rows at a time.
    # Three independent places agree on this:
    #   - gem5 func_unit.hh: saSize = width + height - 1, the classic 2D
    #     systolic fill/drain (rows + cols - 1) -- height is a physical axis.
    #   - Spike systolic_array.cc: one weight deque PER LANE (per output column
    #     n), each holding that column's K weights -- so column depth == K.
    #   - MLIR pass: SYSTOLIC_K_DEPTH defaults to SYSTOLIC_SIZE (the square
    #     array's height), i.e. height is what K is tiled by.
    #
    # Config field names:
    #   pe_M  = array HEIGHT. Despite the name this is the K (reduction) axis,
    #           NOT the token/M axis.        [internal: systolic_array_height]
    #   pe_N  = array WIDTH = output-channel N [internal: vpu_num_lanes / Sn]
    #   pe_P  = 3D depth Sk: extra K values each PE reduces per cycle, via a
    #           per-PE multiplier + adder tree [internal: systolic_array_size_k]
    #   simd_K   = 1D SIMD lane count
    #   simd_bit = 1D SIMD per-lane bit width [internal: vpu_vector_length_bits]
    #
    # Both pe_M and pe_P therefore feed the K axis: K reduced per array pass
    # = pe_M * pe_P (height x depth). That is why the MLIR K-tiling divisor is
    # Sm*Sk -- it is not a hack or an unexplained leak, it is the K capacity of
    # the array.
    #
    # NOTE: a "Sm x Sn x Sk = M x N x K" array (2D face = M x N, depth = K) is a
    # DIFFERENT machine -- that is output-stationary, and would require changing
    # gem5's saSize formula, the MLIR dataflow, and Spike. Not what is built here.
    #
    # Old field names are still accepted as fallback.
    if name == "vpu_num_lanes":                          # PE array width Sn (= pe_N)
        return config_yaml.get("pe_N", config_yaml.get("vpu_num_lanes"))
    if name == "vpu_simd_lanes":                         # 1D SIMD lane count (K)
        return config_yaml.get("simd_K", config_yaml.get("pe_N", config_yaml.get("vpu_num_lanes")))
    if name == "CONFIG_SPAD_INFO":
        # Check if IMEM/WMEM/OMEM are configured
        if "imem_num_banks" in config_yaml:
            imem_size = config_yaml["imem_num_banks"] * (config_yaml["imem_sram_bitwidth"] // 8) * config_yaml["imem_sram_depth"]
            wmem_size = config_yaml["wmem_num_banks"] * (config_yaml["wmem_sram_bitwidth"] // 8) * config_yaml["wmem_sram_depth"]
            omem_size = config_yaml["omem_num_banks"] * (config_yaml["omem_sram_bitwidth"] // 8) * config_yaml["omem_sram_depth"]

            sections_total = imem_size + wmem_size + omem_size
            min_size = 2 * sections_total
            spad_size_bytes = 1
            while spad_size_bytes < min_size:
                spad_size_bytes <<= 1

            imem_vaddr = 0xD0000000
            wmem_vaddr = imem_vaddr + imem_size
            omem_vaddr = wmem_vaddr + wmem_size

            return {
              "spad_vaddr" : imem_vaddr,
              "spad_paddr" : 0x2000000000,
              "spad_size" : spad_size_bytes,
              "spad_total_size" : sections_total,
              "imem_vaddr" : imem_vaddr,
              "imem_size" : imem_size,
              "wmem_vaddr" : wmem_vaddr,
              "wmem_size" : wmem_size,
              "omem_vaddr" : omem_vaddr,
              "omem_size" : omem_size
            }
        else:
            # Fallback to unified SRAM mode
            return {
              "spad_vaddr" : 0xD0000000,
              "spad_paddr" : 0x2000000000,
              "spad_size" : config_yaml["vpu_spad_size_kb_per_lane"] << 10
            }

    if name == "CONFIG_NUM_CORES":
        return config_yaml["num_cores"]
    if name == "vpu_vector_length_bits":                 # 1D SIMD per-lane bit width (= simd_bit)
        return config_yaml.get("simd_bit", config_yaml.get("vpu_vector_length_bits"))

    if name == "pytorchsim_functional_mode":
        return config_yaml['pytorchsim_functional_mode']
    if name == "pytorchsim_timing_mode":
        return config_yaml['pytorchsim_timing_mode']

    # Dataflow. This decides what the array's axes MEAN, so read it before
    # interpreting pe_M below. See rect_array_work/OS_3D_CONTRACT.md.
    #   "ws" (default, legacy): weight-stationary. Stationary tile is K x N, so
    #     the array face is height=K by width=N and M is streamed over time.
    #     pe_M is therefore the K axis, and K per array pass = pe_M * pe_P.
    #   "os": output-stationary 3D. Array face is height=M by width=N, each PE
    #     owns the accumulator for one C[m][n], and pe_P=Sk is the depth (K
    #     values reduced per PE per cycle). Both operands stream; K is tiled by
    #     Sk alone. Here pe_M finally means what its name says.
    if name == "systolic_dataflow":
        df = str(config_yaml.get("dataflow", "ws")).lower()
        assert df in ("ws", "os"), f"Invalid dataflow {df!r}: expected 'ws' or 'os'"
        return df

    # 3D depth Sk (= pe_P): how many K values each PE reduces per cycle, via a
    # per-PE multiplier + adder tree. Costs ceil(log2(Sk)) cycles of adder-tree
    # latency in gem5 (func_unit.hh). In "ws" it multiplies the array's K
    # capacity on top of the height; in "os" it IS the K tiling granularity.
    if name == "systolic_array_size_k":                  # 3D depth Sk (= pe_P)
        return config_yaml.get("pe_P", config_yaml.get("systolic_array_size_k", 1))

    # Array HEIGHT (= pe_M). In "ws" this is the K/reduction axis despite its
    # name; in "os" it is the M/token axis. Defaults to the width N (square
    # array) when not given.
    if name == "systolic_array_height":                  # array height
        _N = config_yaml.get("pe_N", config_yaml.get("vpu_num_lanes"))
        return config_yaml.get("pe_M", config_yaml.get("systolic_array_height", _N))

    # Rectangular/3D PE array modelling mode.
    #   False: only valid when the array is actually square (pe_M == pe_N) and
    #     has no depth (pe_P == 1) -- gem5 measures that square array directly,
    #     no correction needed.
    #   True (auto-enabled whenever pe_M != pe_N or pe_P > 1): Sm*Sk is
    #     threaded into the MLIR passes (K tiled by Sm*Sk, see
    #     systolic_array_size_k comment above) and Sk into gem5
    #     (--vlane-depth, Sk adder-tree layers) so the rectangular/3D array is
    #     simulated directly. Requires mlir-opt built from
    #     patches/llvm-project.patch and gem5 built from patches/gem5.patch
    #     (mount both into the docker container -- see
    #     rect_array_work_scripts/run_cyc.sh's MLIROPT/GEM5 env vars).
    if name == "systolic_array_real_rect":
        # Explicit override wins; otherwise auto-enable whenever the PE array
        # geometry is non-square (pe_M != pe_N), has depth (pe_P > 1), or runs
        # the output-stationary dataflow (which always needs the real path).
        if "systolic_array_real_rect" in config_yaml:
            return bool(config_yaml["systolic_array_real_rect"])
        if str(config_yaml.get("dataflow", "ws")).lower() == "os":
            return True
        _N = config_yaml.get("pe_N", config_yaml.get("vpu_num_lanes"))
        _M = config_yaml.get("pe_M", config_yaml.get("systolic_array_height", _N))
        _P = config_yaml.get("pe_P", config_yaml.get("systolic_array_size_k", 1))
        return (_M != _N) or (_P > 1)

    # Mapping strategy
    if name == "codegen_mapping_strategy":
        codegen_mapping_strategy = config_yaml["codegen_mapping_strategy"]
        assert(codegen_mapping_strategy in ["heuristic", "autotune", "external-then-heuristic", "external-then-autotune"]), "Invalid mapping strategy!"
        return codegen_mapping_strategy

    if name == "codegen_external_mapping_file":
        return config_yaml["codegen_external_mapping_file"]

    # Autotune config
    if name == "codegen_autotune_max_retry":
        return config_yaml["codegen_autotune_max_retry"]
    if name == "codegen_autotune_template_topk":
        return config_yaml["codegen_autotune_template_topk"]
    # Added to first candidate wall time for other candidates' TOGSim subprocess timeout (>= 1 s).
    if name == "codegen_autotune_wall_slack_sec":
        v = float(config_yaml.get("codegen_autotune_wall_slack_sec", 15))
        return max(1.0, v)

    # Compiler Optimization
    if name == "codegen_compiler_optimization":
        opt_level = config_yaml["codegen_compiler_optimization"]
        valid_opts = {
            "fusion",
            "reduction_epilogue",
            "reduction_reduction",
            "prologue",
            "single_batch_conv",
            "multi_tile_conv",
            "subtile"
        }
        if opt_level == "all" or opt_level == "none":
            pass
        elif isinstance(opt_level, list):
            # Check if provided list contains only valid options
            invalids = set(opt_level) - valid_opts
            assert not invalids, f"Invalid optimization options found: {invalids}"
        else:
            assert False, "Invalid format: Must be 'all', none, or a list of options."
        return opt_level

    # Advanced fusion options
    is_opt_enabled = lambda key: (__getattr__("codegen_compiler_optimization") == "all") or \
                                 (isinstance(__getattr__("codegen_compiler_optimization"), list) and \
                                  key in __getattr__("codegen_compiler_optimization"))
    if name == "CONFIG_FUSION":
        return is_opt_enabled("fusion")
    if name == "CONFIG_FUSION_REDUCTION_EPILOGUE":
        return is_opt_enabled("reduction_epilogue") # Fixed typo here as well
    if name == "CONFIG_FUSION_REDUCTION_REDUCTION":
        return is_opt_enabled("reduction_reduction")
    if name == "CONFIG_FUSION_PROLOGUE":
        return is_opt_enabled("prologue")
    if name == "CONFIG_SINGLE_BATCH_CONV":
        return is_opt_enabled("single_batch_conv")
    if name == "CONFIG_MULTI_TILE_CONV":
        return is_opt_enabled("multi_tile_conv")
    if name == "CONFIG_SUBTILE":
        return is_opt_enabled("subtile")

    if name == "CONFIG_TOGSIM_DEBUG_LEVEL":
        return os.environ.get("TOGSIM_DEBUG_LEVEL", "")
    if name == "CONFIG_TORCHSIM_DUMP_PATH":
        dump_path = os.environ.get('TORCHSIM_DUMP_PATH', default = os.path.join(CONFIG_TORCHSIM_DIR, "outputs"))
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(dump_path, ".torchinductor")
        return dump_path
    if name == "CONFIG_TORCHSIM_LOG_PATH":
        return os.environ.get('TORCHSIM_LOG_PATH', default = os.path.join(CONFIG_TORCHSIM_DIR, "togsim_results"))

# SRAM Buffer allocation plan
def load_plan_from_module(module_path):
    if module_path is None:
      return None

    try:
        spec = importlib.util.spec_from_file_location("plan_module", module_path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'plan'):
            return module.plan
        return None
    except Exception as e:
        print(f"[Warning] Failed to load SRAM buffer plan from module: {e}")
        return None

CONFIG_SRAM_BUFFER_PLAN_PATH = os.environ.get("SRAM_BUFFER_PLAN_PATH", default=None)
CONFIG_SRAM_BUFFER_PLAN = load_plan_from_module(CONFIG_SRAM_BUFFER_PLAN_PATH)

# For ILS experiment
CONFIG_TLS_MODE = int(os.environ.get('TORCHSIM_TLS_MODE', default=1))

CONFIG_USE_TIMING_POOLING = int(os.environ.get('TORCHSIM_USE_TIMING_POOLING', default=0))

CONFIG_DEBUG_MODE = int(os.environ.get('TORCHSIM_DEBUG_MODE', default=0))


def setup_logger(name=None, level=None):
    """
    Setup a logger with consistent formatting across all modules.

    Args:
        name: Logger name (default: __name__ of calling module)
        level: Logging level (default: DEBUG if CONFIG_DEBUG_MODE else INFO)

    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        # Get the calling module's name
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'PyTorchSim')

    # Convert logger name to lowercase
    name = name.lower()
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Always output to stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # ALSO output to log file in togsim_results
        try:
            log_dir = os.environ.get('TORCHSIM_LOG_PATH',
                                    default=os.path.join(CONFIG_TORCHSIM_DIR, "togsim_results"))
            os.makedirs(log_dir, exist_ok=True)

            # Use a fixed log filename for each run (based on timestamp)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            import random
            rand_suffix = ''.join(random.choices('0123456789abcdef', k=8))
            log_filename = os.path.join(log_dir, f"{timestamp}_{rand_suffix}.log")

            file_handler = logging.FileHandler(log_filename, mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"[WARNING] Failed to setup file logging: {e}")

        # Set log level
        if level is None:
            level = logging.DEBUG if CONFIG_DEBUG_MODE else logging.INFO
        logger.setLevel(level)

    return logger