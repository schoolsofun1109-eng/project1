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

def get_new_log_file(before_snapshot, after_snapshot=None, max_retries=20, retry_delay=0.5, min_size=1000):
    """Find the log file modified between two snapshots.

    Returns path to the log file that was created/modified with min_size bytes, or None if not found.
    Waits for file to have sufficient content (Spike simulation completion).
    """
    log_dir = get_log_dir()

    # First pass: find newly created files
    for attempt in range(max_retries):
        try:
            if not os.path.exists(log_dir):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
            import glob
            current_files = glob.glob(os.path.join(log_dir, "*.log"))
            after_snapshot = {}
            for f in current_files:
                try:
                    after_snapshot[f] = os.path.getmtime(f)
                except OSError:
                    pass

            # Find files modified/created between snapshots
            modified_files = []
            for f_after, mtime_after in after_snapshot.items():
                mtime_before = before_snapshot.get(f_after)
                if mtime_before is None:
                    # New file created
                    modified_files.append((f_after, mtime_after))
                elif mtime_after > mtime_before:
                    # File modified
                    modified_files.append((f_after, mtime_after))

            if modified_files:
                # Pick the most recently modified file
                modified_log = max(modified_files, key=lambda x: x[1])[0]

                # Wait for file to have sufficient content (Spike completion)
                for size_check in range(10):
                    try:
                        file_size = os.path.getsize(modified_log)
                        if file_size >= min_size:
                            return modified_log
                    except OSError:
                        pass

                    if size_check < 9:
                        time.sleep(0.2)

                # If size check times out, still return the file (it may have metrics)
                return modified_log

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
    """Compile a function with torch.compile using snapshot-based log file detection.

    Returns: (compiled_fn, log_file_path)
    This uses before/after snapshots to reliably detect log file changes.
    """
    # Snapshot current log files BEFORE torch.compile
    before_snapshot = snapshot_log_files()

    try:
        # Compile the function
        compiled_fn = torch.compile(dynamic=False)(fn)

        # Give filesystem time to sync
        time.sleep(0.5)

        # Find the log file that was created/modified
        log_file = get_new_log_file(before_snapshot, max_retries=15, retry_delay=0.3)

        return compiled_fn, log_file

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

                    systolic_matches = re.findall(r'Systolic array \[0\] utilization\(%\): ([\d.]+)', content)
                    bw_matches = re.findall(r'channels 0\.\.15 combined.*?(\d+\.?\d*) GB/s', content)

                    if systolic_matches:
                        log_result(f"Systolic util: {systolic_matches[-1]}%")
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

    # Compile with snapshot-based log file detection
    opt_fn, log_file = compile_with_stdout_capture(custom_matmul)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    log_result(f"[DEBUG] torch.compile completed, waiting for log file...")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    res = opt_fn(x1, w1)
    y = custom_matmul(x2, w2)
    test_result("Matmul Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

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

    # Compile with snapshot-based log file detection
    opt_fn, log_file = compile_with_stdout_capture(custom_matmul)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    log_result(f"[DEBUG] torch.compile completed, waiting for log file...")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    res = opt_fn(b1, x1, w1)
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

    # Compile with snapshot-based log file detection
    opt_fn, log_file = compile_with_stdout_capture(custom_matmul)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    log_result(f"[DEBUG] torch.compile completed, waiting for log file...")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    res = opt_fn(b1, x1, w1)
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

    # Compile with snapshot-based log file detection
    opt_fn, log_file = compile_with_stdout_capture(custom_linear)

    if opt_fn is None:
        log_result(f"[ERROR] Failed to compile function")
        return

    log_result(f"[DEBUG] torch.compile completed, waiting for log file...")
    if log_file:
        log_result(f"[DEBUG] Log file: {log_file}")

    res = opt_fn(x1, w1, b1)
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
