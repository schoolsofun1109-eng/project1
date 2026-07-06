import os
import sys

# Disable auto-loading of extension backends before importing torch
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch
import re
import time
from datetime import datetime

# Initialize config paths at module load time
sys.path.insert(0, '/workspace/PyTorchSim')

# CRITICAL: Register npu device and backend (required for torch.compile to use npu)
# Try multiple import paths (user's local code, container's installed package, etc.)
npu_registered = False
for import_path in [
    'PyTorchSimDevice.torch_openreg',  # From /workspace or installed package
]:
    try:
        __import__(import_path)
        print(f"[INFO] NPU device registered via {import_path}")
        npu_registered = True
        break
    except Exception as e:
        print(f"[INFO] Could not register npu via {import_path}: {e}")

if not npu_registered:
    print("[WARNING] NPU device not registered - tests will run on CPU")

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

def get_log_file_after_timestamp(start_time, max_retries=20, retry_delay=0.5):
    """Find log file modified after start_time (timestamp-based detection).

    This is the most reliable method: it finds any file whose mtime >= start_time,
    which is guaranteed to be modified after the test started.

    Args:
        start_time: Unix timestamp when the test execution started
        max_retries: Number of times to retry if no file found
        retry_delay: Delay between retries

    Returns:
        Path to the log file modified after start_time, or None if not found.
    """
    log_dir = get_log_dir()

    initial_sizes = {}

    for attempt in range(max_retries):
        try:
            if not os.path.exists(log_dir):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

            import glob
            log_files = glob.glob(os.path.join(log_dir, "*.log"))

            # On first attempt, record initial sizes
            if not initial_sizes:
                for f in log_files:
                    try:
                        initial_sizes[f] = os.path.getsize(f)
                    except OSError:
                        pass

            # Find files that grew (size increased) since start_time
            growing_files = []
            for f in log_files:
                try:
                    current_size = os.path.getsize(f)
                    initial_size = initial_sizes.get(f, current_size)
                    mtime = os.path.getmtime(f)

                    # File grew OR mtime is after start_time
                    if current_size > initial_size or mtime >= start_time:
                        growing_files.append((f, current_size, mtime))
                except OSError:
                    pass

            if growing_files:
                # Pick file with most recent mtime
                latest_log = max(growing_files, key=lambda x: x[2])[0]
                size_change = os.path.getsize(latest_log) - initial_sizes.get(latest_log, 0)
                print(f"[LOG_DETECT] ✅ Found: {os.path.basename(latest_log)} (size +{size_change} bytes, mtime={os.path.getmtime(latest_log):.2f})")
                return latest_log

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        except Exception as e:
            print(f"[LOG_DETECT] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    print(f"[LOG_DETECT] ❌ No log file found after {start_time:.2f} (after {max_retries} retries)")
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

                    systolic_match = re.search(r'Systolic array \[0\] utilization\(%\): ([\d.]+)', content)
                    bw_match = re.search(r'channels 0\.\.15 combined.*?(\d+\.?\d*) GB/s', content)

                    if systolic_match:
                        log_result(f"Systolic util: {systolic_match.group(1)}%")
                    if bw_match:
                        log_result(f"DRAM BW: {bw_match.group(1)} GB/s")
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
    """Test matrix multiplication with timestamp-based log file detection."""
    def custom_matmul(a, b):
        return torch.matmul(a, b)

    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")

    # Removed timestamp tracking - use log file content instead
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(x1, w1)

    log_file = None  # Let test_result fallback to get_latest_log_file

    y = custom_matmul(x2, w2)
    test_result("Matmul Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

def test_addmm(device, input_size=128, hidden_size=128, output_size=128, bias_rank=1):
    """Test addmm operation with snapshot-based log tracking."""
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

    # Removed timestamp tracking - use log file content instead
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)

    log_file = None  # Let test_result fallback to get_latest_log_file

    y = custom_matmul(b2, x2, w2)
    test_result("Addmm Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

def test_addmm2(device, input_size=128, hidden_size=128, output_size=128):
    """Test matmul operation variant with snapshot-based log tracking."""
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

    # Removed timestamp tracking - use log file content instead
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)

    log_file = None  # Let test_result fallback to get_latest_log_file

    y = custom_matmul(b2, x2, w2)
    test_result("Addmm2 Forward", res, y, m=input_size, n=output_size, k=hidden_size, log_file=log_file)

def test_linear(device, input_size=128, hidden_size=128, output_size=128):
    """Test linear layer operation with timestamp-based log file detection."""
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

    # Removed timestamp tracking - use log file content instead
    opt_fn = torch.compile(dynamic=False)(custom_linear)
    res = opt_fn(x1, w1, b1)

    log_file = None  # Let test_result fallback to get_latest_log_file

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
