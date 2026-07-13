import os
import sys

# Disable auto-loading of extension backends before importing torch
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch
import re
import time
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout

# Initialize config paths at module load time
sys.path.insert(0, '/workspace/PyTorchSim')

# CRITICAL: Register npu device and backend (required for torch.compile to use npu)
# Try multiple import paths (user's local code, container's installed package, etc.)
print("[STARTUP] Attempting to register NPU device...")
npu_registered = False
for import_path in [
    'PyTorchSimDevice.torch_openreg',  # From /workspace or installed package
]:
    try:
        print(f"[STARTUP] Trying import path: {import_path}")
        __import__(import_path)
        print(f"[STARTUP] ✓ NPU device registered via {import_path}")
        npu_registered = True
        break
    except Exception as e:
        print(f"[STARTUP] ✗ Could not register npu via {import_path}: {e}")

if npu_registered:
    print("[STARTUP] ✓ NPU device is available")
    available_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[STARTUP] CUDA available: {torch.cuda.is_available()}, count: {available_devices}")
    try:
        npu_dev = torch.device('npu:0')
        print(f"[STARTUP] ✓ NPU device object created: {npu_dev}")
    except Exception as e:
        print(f"[STARTUP] ✗ Failed to create npu:0 device: {e}")
else:
    print("[STARTUP] ✗ NPU device NOT registered - tests will run on CPU")

try:
    from PyTorchSimFrontend import extension_config
    LOG_DIR = extension_config.CONFIG_TORCHSIM_LOG_PATH
except Exception:
    LOG_DIR = "/workspace/PyTorchSim/togsim_results"

# Global results list
RESULTS = []
RESULTS_FILE = None
LAST_LOG_FILE = None

def get_log_dir():
    """Get the log directory with multiple fallbacks."""
    log_dir = os.environ.get('TORCHSIM_LOG_PATH')
    if not log_dir:
        try:
            sys.path.insert(0, '/workspace/PyTorchSim')
            from PyTorchSimFrontend import extension_config
            log_dir = extension_config.CONFIG_TORCHSIM_LOG_PATH
        except:
            log_dir = "/workspace/PyTorchSim/togsim_results"
    return log_dir

def init_results_file():
    global RESULTS_FILE
    results_dir = LOG_DIR
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_FILE = os.path.join(results_dir, f"test_results_{timestamp}.txt")
    with open(RESULTS_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PYTORCH SIM TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")

def log_result(text):
    global RESULTS_FILE, RESULTS
    RESULTS.append(text)
    if RESULTS_FILE:
        with open(RESULTS_FILE, 'a') as f:
            f.write(text + "\n")
    print(text)

def snapshot_log_files():
    """Create a snapshot of current log files with their modification times.
    Returns a dict mapping log file paths to their mtime values.
    """
    log_dir = get_log_dir()
    try:
        if os.path.exists(log_dir):
            import glob
            log_files = glob.glob(os.path.join(log_dir, "*.log"))
            snapshot = {}
            for f in log_files:
                try:
                    mtime = os.path.getmtime(f)
                    snapshot[f] = mtime
                except OSError:
                    pass
            return snapshot
    except Exception as e:
        pass
    return {}

def _log_has_metrics(path):
    """Return True if the log file contains completed TOGSim metrics."""
    try:
        with open(path, 'r') as f:
            content = f.read()
        return 'Total execution cycles' in content
    except Exception:
        return False

def get_new_log_file(before_snapshot, after_snapshot=None, max_retries=20, retry_delay=0.5, min_size=1000):
    """Find the NEW log file (created/modified after before_snapshot) that contains
    completed TOGSim metrics.

    A single opt_fn() execution may create multiple log files (e.g. empty Gem5
    stub files plus the real metrics file). We therefore prefer, among the newly
    created/modified files, the one that actually contains 'Total execution cycles'.
    We retry to give the (synchronous but file-buffered) TOGSim subprocess time to
    flush its output to disk.
    """
    import glob
    log_dir = get_log_dir()

    for attempt in range(max_retries):
        try:
            if not os.path.exists(log_dir):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

            current_files = glob.glob(os.path.join(log_dir, "*.log"))

            # Collect files created/modified since the before-snapshot
            modified_files = []
            for f in current_files:
                try:
                    mtime_after = os.path.getmtime(f)
                except OSError:
                    continue
                mtime_before = before_snapshot.get(f)
                if mtime_before is None or mtime_after > mtime_before:
                    modified_files.append((f, mtime_after))

            if modified_files:
                # Prefer newest files first, but only accept ones with real metrics.
                modified_files.sort(key=lambda x: x[1], reverse=True)

                # 1) Best: a new file that already contains completed metrics
                for f, _ in modified_files:
                    if _log_has_metrics(f):
                        return f

                # 2) Otherwise, keep waiting for metrics to be flushed
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

                # 3) Last resort on final attempt: return the largest new file
                #    that meets the size threshold (may still be parseable)
                sized = [(f, os.path.getsize(f)) for f, _ in modified_files
                         if os.path.exists(f)]
                sized = [(f, s) for f, s in sized if s >= min_size]
                if sized:
                    return max(sized, key=lambda x: x[1])[0]
                return modified_files[0][0]

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return None

def get_log_file_after_timestamp(start_time, max_retries=60, retry_delay=0.2):
    """Find log file created/modified AFTER test start_time.

    Simple & precise: find files with metrics (>500 bytes) AND mtime >= start_time - 1.0
    """
    log_dir = get_log_dir()
    import glob

    # Allow 1 second slack for clock differences
    time_threshold = start_time - 1.0

    for attempt in range(max_retries):
        try:
            if not os.path.exists(log_dir):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

            log_files = glob.glob(os.path.join(log_dir, "*.log"))

            # Find files: WITH metrics AND created/modified after test started
            candidate_files = []
            for f in log_files:
                try:
                    size = os.path.getsize(f)
                    mtime = os.path.getmtime(f)
                    # File must have metrics (>500 bytes) AND be created after test start
                    if size >= 500 and mtime >= time_threshold:
                        candidate_files.append((f, mtime))
                except OSError:
                    pass

            if candidate_files:
                # Pick the MOST RECENT file (most recent mtime)
                latest_log = max(candidate_files, key=lambda x: x[1])[0]
                return latest_log

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return None

def extract_cycle_from_log(log_file_path):
    """Extract the LAST 'Total execution cycles' from TOGSim log file.

    Since tests append to the same log file, we need the last (most recent) cycle value.
    """
    try:
        if not os.path.exists(log_file_path):
            return None
        with open(log_file_path, 'r') as f:
            content = f.read()
            matches = re.findall(r'Total execution cycles:\s*(\d+)', content)
            if matches:
                return int(matches[-1])
    except Exception as e:
        pass
    return None

def find_togsim_log_in_stdout(stdout_content):
    """Parse TOGSim log file path from stdout.

    TOGSim outputs: "[TOGSim] Simulation log is stored to "/path/to/YYYYMMDD_HHMMSS_XXXXXXXX.log""
    """
    match = re.search(r'\[TOGSim\] Simulation log is stored to "([^"]+)"', stdout_content)
    if match:
        return match.group(1)
    return None

def compile_with_stdout_capture(fn):
    """Compile a function with torch.compile and snapshot log files BEFORE execution.

    Returns: (compiled_fn, before_snapshot)

    IMPORTANT: torch.compile() is LAZY - it only compiles, it does NOT execute.
    The actual TOGSim simulation (and log file writing) happens when the compiled
    function is CALLED (e.g. opt_fn(x1, w1)). Therefore we must NOT detect the log
    file here. Instead we return the before-execution snapshot, and the caller must
    call get_new_log_file(before_snapshot) AFTER invoking the compiled function.
    """
    # Snapshot current log files BEFORE compile+execution.
    # torch.compile itself does not run the simulation, so this snapshot correctly
    # captures the pre-existing log files. The new log appears only after opt_fn().
    before_snapshot = snapshot_log_files()

    try:
        compiled_fn = torch.compile(dynamic=False)(fn)
        return compiled_fn, before_snapshot
    except Exception as e:
        log_result(f"[ERROR] torch.compile failed: {e}")
        return None, None

def get_latest_log_file(max_retries=5, retry_delay=0.2):
    """DEPRECATED: Use snapshot_log_files() and get_new_log_file() instead.

    This function is kept for backward compatibility but should not be used
    for new code. The snapshot-based approach is more reliable and doesn't
    suffer from race conditions with modification times.
    """
    global LAST_LOG_FILE

    log_dir = get_log_dir()

    for attempt in range(max_retries):
        try:
            if not os.path.exists(log_dir):
                if attempt == max_retries - 1:
                    return None
                time.sleep(retry_delay)
                continue

            import glob
            log_files = glob.glob(os.path.join(log_dir, "*.log"))
            if log_files:
                latest = max(log_files, key=os.path.getmtime)
                LAST_LOG_FILE = latest
                return latest

            # No files found yet, retry if not last attempt
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(retry_delay)

    return None

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4, m=None, n=None, k=None, log_file=None):
    """Verify test result and extract simulation metrics from log file.

    Args:
        name: Test name for output
        out: Output from NPU computation
        cpu_out: Output from CPU computation
        rtol: Relative tolerance for allclose check
        atol: Absolute tolerance for allclose check
        m, n, k: Matrix dimensions for logging
        log_file: Path to the TOGSim log file for this specific test.
                  Should be obtained from get_new_log_file() after torch.compile.
    """
    global LAST_LOG_FILE

    # Calculate differences for verification
    diff_detail = (out.cpu() - cpu_out).abs()
    rel_diff = diff_detail / (cpu_out.abs() + 1e-8)
    max_abs_diff = diff_detail.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = diff_detail.mean().item()

    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        log_result("-" * len(message))
        log_result(message)
        log_result("-" * len(message))

        # Print matrix dimensions and input values
        if m is not None and n is not None and k is not None:
            log_result(f"Matrix shape: {m}×{k} × {k}×{n} → {m}×{n}")

            # Print sample input values
            log_result(f"Input (X) first 3 values: {out.cpu().flatten()[:3].tolist()}")
            log_result(f"CPU  (Y) first 3 values: {cpu_out.flatten()[:3].tolist()}")

            # Compare CPU vs NPU outputs
            npu_sum = out.sum().item()
            cpu_sum = cpu_out.sum().item()
            log_result(f"NPU sum: {npu_sum:.6f}, CPU sum: {cpu_sum:.6f}")

            # Print difference analysis
            log_result("=== Precision Analysis ===")
            log_result(f"Max abs diff: {max_abs_diff:.6e}")
            log_result(f"Max rel diff: {max_rel_diff:.6e}")
            log_result(f"Mean abs diff: {mean_abs_diff:.6e}")

        # Extract performance metrics from TOGSim log
        log_result("")
        log_result("=== SIMULATION METRICS ===")

        # If no log file provided, fall back to deprecated get_latest_log_file
        if log_file is None:
            # Wait for log file to be written (test execution + file I/O)
            time.sleep(1.0)
            log_file = get_latest_log_file(max_retries=5, retry_delay=0.2)

        if log_file:
            log_result(f"Log: {os.path.basename(log_file)}")

            # Update tracked log file
            LAST_LOG_FILE = log_file

            # Extract cycle count
            cycles = extract_cycle_from_log(log_file)
            if cycles is not None:
                log_result(f"Cycles: {cycles}")
            else:
                log_result("(Cycles: not found in log)")

            # Try to extract other metrics
            try:
                with open(log_file, 'r') as f:
                    content = f.read()

                    # A core can have multiple systolic arrays (systolic_arrays_per_core).
                    # Small workloads may run entirely on one array (e.g. array [1]),
                    # leaving array [0] at 0% active_cycles. Reading only array [0] would
                    # then wrongly report 0.00%. So we capture ALL arrays from the final
                    # stats block and average their utilization.
                    array_matches = re.findall(
                        r'Systolic array \[(\d+)\] utilization\(%\): ([\d.]+)', content)
                    bw_matches = re.findall(r'channels 0\.\.15 combined.*?(\d+\.?\d*) GB/s', content)

                    if array_matches:
                        # Determine how many arrays per core (max index + 1),
                        # then take the LAST full set (most recent kernel's stats).
                        num_arrays = max(int(idx) for idx, _ in array_matches) + 1
                        last_set = array_matches[-num_arrays:]
                        utils = [float(v) for _, v in last_set]
                        avg_util = sum(utils) / len(utils)
                        # Report each array individually AND the average
                        for idx, v in last_set:
                            log_result(f"Systolic array [{idx}] util: {v}%")
                        log_result(f"Systolic util (avg of {num_arrays} arrays): {avg_util:.2f}%")
                    if bw_matches:
                        log_result(f"DRAM BW: {bw_matches[-1]} GB/s")
            except Exception as e:
                log_result(f"(Metrics extraction error: {str(e)})")
        else:
            log_result("(No log file found - test may not have executed)")

        log_result("")

    else:
        message = f"|{name} Test Failed|"
        log_result("-" * len(message))
        log_result(message)
        log_result("-" * len(message))
        log_result(f"Shape: NPU {out.shape} vs CPU {cpu_out.shape}")
        log_result("custom out: " + str(out.cpu()))
        log_result("cpu out: " + str(cpu_out))
        log_result(f"Results saved to: {RESULTS_FILE}")
        exit(1)

def test_matmul(device, input_size=128, hidden_size=128, output_size=128):
    """Test matrix multiplication with snapshot-based log file detection."""
    def custom_matmul(a, b):
        return torch.matmul(a, b)

    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")

    log_result(f"\n[DEBUG] test_matmul({input_size}x{hidden_size}x{output_size}) START (t={time.time()})")

    # Compile (lazy) and snapshot log files BEFORE execution
    opt_fn, before_snapshot = compile_with_stdout_capture(custom_matmul)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    # ACTUAL execution: this is when TOGSim runs and writes the log file
    res = opt_fn(x1, w1)

    # Detect the log file created by THIS execution (after opt_fn, with size wait)
    log_file = get_new_log_file(before_snapshot, max_retries=20, retry_delay=0.5)
    log_result(f"[DEBUG] torch.compile+exec completed")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    y = custom_matmul(x2, w2)
    test_result("Matmul Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

def test_matmul_golden(device):
    """Hand-verifiable golden case for SA/memory validation.

    A(2x4) @ B(4x2) with fixed integer values:
        A = [[1,2,3,4],
             [5,6,7,8]]
        B = [[1,2],[3,4],[5,6],[7,8]]
        A@B = [[  1+6+15+28 ,  2+8+18+32 ],   = [[ 50,  60],
               [ 5+18+35+56, 10+24+42+64]]      [114, 140]]
    Run with SPIKE_DEBUG=1 | tileflow.py to check the DRAIN (SA output) values
    equal 50/60/114/140 by hand.
    """
    def custom_matmul(a, b):
        return torch.matmul(a, b)

    A = torch.tensor([[1., 2., 3., 4.],
                      [5., 6., 7., 8.]])
    B = torch.tensor([[1., 2.],
                      [3., 4.],
                      [5., 6.],
                      [7., 8.]])
    expected = A @ B
    print("=================== GOLDEN A(2x4) @ B(4x2) ===================")
    print("A =", A.tolist())
    print("B =", B.tolist())
    print("HAND/CPU expected =", expected.tolist(), " -> [[50,60],[114,140]]")

    x1, w1 = A.to(device), B.to(device)
    opt_fn, before_snapshot = compile_with_stdout_capture(custom_matmul)
    if opt_fn is None:
        print("[ERROR] Failed to compile function")
        return
    res = opt_fn(x1, w1)
    npu = res.cpu()
    print("NPU out =", npu.tolist())
    match = torch.allclose(npu, expected, atol=1e-2)
    print("GOLDEN MATCH =", bool(match))
    test_result("Matmul Golden", res, expected, m=2, n=2, k=4)

def test_golden_cases(device, case=0):
    """Hand-verifiable golden matmul cases for SA/dataflow validation.

    Run with SPIKE_DEBUG=1 | tileflow.py to see the DRAM->SRAM->SA->DRAM flow
    and check each stage's values by hand. Cases with K>128 exercise multiple
    K-passes (the 128-tall array reduces K in 128-chunks).
    """
    def mm(a, b):
        return torch.matmul(a, b)
    cases = [
        # name, A, B  (all hand-verifiable)
        ("2x4 @ 4x2  (K=4, single pass)",
         torch.tensor([[1.,2,3,4],[5,6,7,8]]), torch.tensor([[1.,2],[3,4],[5,6],[7,8]])),
        ("2x8 @ 8x2  (all 2 * all 3 -> 8*6=48, K=8)",
         torch.full((2,8),2.), torch.full((8,2),3.)),
        ("2x128 @ 128x2  (ones -> 128, K=128 = exactly 1 full pass)",
         torch.ones(2,128), torch.ones(128,2)),
        ("2x256 @ 256x2  (ones -> 256, K=256 = 2 K-passes, accumulate)",
         torch.ones(2,256), torch.ones(256,2)),
        ("2x384 @ 384x2  (ones -> 384, K=384 = 3 K-passes)",
         torch.ones(2,384), torch.ones(384,2)),
    ]
    name, A, B = cases[case]
    exp = A @ B
    print(f"=============== GOLDEN CASE {case}: {name} ===============")
    print(f"  A shape {tuple(A.shape)}, B shape {tuple(B.shape)}, K={A.shape[1]}, K-passes(128-array)={-(-A.shape[1]//128)}")
    print(f"  HAND/CPU expected row0 = {exp[0].tolist()[:4]}{' ...' if exp.shape[1]>4 else ''}")
    opt_fn, _ = compile_with_stdout_capture(mm)
    if opt_fn is None:
        print("  [ERROR] compile failed"); return
    res = opt_fn(A.to(device), B.to(device)).cpu()
    print(f"  NPU out row0          = {res[0].tolist()[:4]}{' ...' if res.shape[1]>4 else ''}")
    print(f"  GOLDEN MATCH = {bool(torch.allclose(res, exp, atol=1e-2))}")

def test_addmm(device, input_size=128, hidden_size=128, output_size=128, bias_rank=1):
    """Test addmm operation with stdout-based log tracking."""
    def custom_matmul(bias, a, b):
        return torch.addmm(bias, a, b)

    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(output_size) if bias_rank == 1 else torch.randn(input_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")

    log_result(f"\n[DEBUG] test_addmm({input_size}x{hidden_size}x{output_size}) START (t={time.time()})")

    # Compile (lazy) and snapshot log files BEFORE execution
    opt_fn, before_snapshot = compile_with_stdout_capture(custom_matmul)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    # ACTUAL execution: this is when TOGSim runs and writes the log file
    res = opt_fn(b1, x1, w1)

    # Detect the log file created by THIS execution (after opt_fn, with size wait)
    log_file = get_new_log_file(before_snapshot, max_retries=20, retry_delay=0.5)
    log_result(f"[DEBUG] torch.compile+exec completed")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    y = custom_matmul(b2, x2, w2)
    test_result("Addmm Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

def test_addmm2(device, input_size=128, hidden_size=128, output_size=128):
    """Test matmul operation variant with stdout-based log tracking."""
    def custom_matmul(bias, a, b):
        return torch.matmul(a, b) #+ bias

    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(input_size, 1, dtype=torch.float32)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")

    log_result(f"\n[DEBUG] test_addmm2({input_size}x{hidden_size}x{output_size}) START (t={time.time()})")

    # Compile (lazy) and snapshot log files BEFORE execution
    opt_fn, before_snapshot = compile_with_stdout_capture(custom_matmul)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    # ACTUAL execution: this is when TOGSim runs and writes the log file
    res = opt_fn(b1, x1, w1)

    # Detect the log file created by THIS execution (after opt_fn, with size wait)
    log_file = get_new_log_file(before_snapshot, max_retries=20, retry_delay=0.5)
    log_result(f"[DEBUG] torch.compile+exec completed")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    y = custom_matmul(b2, x2, w2)
    test_result("Addmm2 Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

def test_linear(device, input_size=128, hidden_size=128, output_size=128):
    """Test linear layer operation with stdout-based log file detection."""
    def custom_linear(a, b, bias):
        linear = torch.nn.Linear(hidden_size, output_size)
        linear.weight = torch.nn.Parameter(b)
        linear.bias = torch.nn.Parameter(bias)
        return linear(a)

    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(output_size, hidden_size)
    bias = torch.randn(output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")

    log_result(f"\n[DEBUG] test_linear({input_size}x{hidden_size}x{output_size}) START (t={time.time()})")

    # Compile (lazy) and snapshot log files BEFORE execution
    opt_fn, before_snapshot = compile_with_stdout_capture(custom_linear)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    # ACTUAL execution: this is when TOGSim runs and writes the log file
    res = opt_fn(x1, w1, b1)

    # Detect the log file created by THIS execution (after opt_fn, with size wait)
    log_file = get_new_log_file(before_snapshot, max_retries=20, retry_delay=0.5)
    log_result(f"[DEBUG] torch.compile+exec completed")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    y = custom_linear(x2, w2, b2)
    test_result("Linear Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

if __name__ == "__main__":
    init_results_file()

    # Print CONFIG information
    try:
        sys.path.insert(0, '/workspace/PyTorchSim')
        from extension_config import CONFIG_SPAD_INFO, CONFIG
        log_result("=" * 80)
        log_result("CONFIGURATION INFO")
        log_result("=" * 80)
        if CONFIG_SPAD_INFO:
            log_result(f"IMEM: vaddr={hex(CONFIG_SPAD_INFO.get('imem_vaddr', 0))}, "
                      f"size={CONFIG_SPAD_INFO.get('imem_size', 0)} bytes")
            log_result(f"WMEM: vaddr={hex(CONFIG_SPAD_INFO.get('wmem_vaddr', 0))}, "
                      f"size={CONFIG_SPAD_INFO.get('wmem_size', 0)} bytes")
            log_result(f"OMEM: vaddr={hex(CONFIG_SPAD_INFO.get('omem_vaddr', 0))}, "
                      f"size={CONFIG_SPAD_INFO.get('omem_size', 0)} bytes")
            log_result(f"Total scratchpad: {CONFIG_SPAD_INFO.get('spad_total_size', 0)} bytes")
            log_result(f"Scratchpad size (Spike): {CONFIG_SPAD_INFO.get('spad_size', 0)} bytes")
        if CONFIG:
            log_result(f"VPU lanes: {CONFIG.get('vpu_num_lanes', 'N/A')}")
        log_result("=" * 80)
        log_result("")
    except Exception as e:
        log_result(f"(Config unavailable: {str(e)})")
        log_result("")

    # Try to use npu device, fall back to cpu if not available
    try:
        device = torch.device("npu:0")
    except RuntimeError:
        print("[WARNING] NPU device not available, using CPU instead")
        device = torch.device("cpu")
    test_matmul(device, 32, 32, 32)
    test_matmul(device, 128, 128, 128)
    test_matmul(device, 256, 256, 256)
    test_matmul(device, 128, 256, 256)
    test_matmul(device, 128, 63, 56)
    test_addmm(device, 128, 256, 512)
    test_addmm(device, 128, 256, 512, bias_rank=2)
    test_addmm(device, 129, 61, 56)
    test_addmm2(device, 129, 61, 56)
    test_addmm(device, 129*4, 61*4, 56*4)
    test_addmm2(device, 129*4, 61*4, 56*4)

    # Print summary
    log_result("=" * 80)
    log_result(f"All tests completed!")
    log_result(f"Results saved to: {RESULTS_FILE}")
    log_result("=" * 80)
