import torch
import os
import re
from datetime import datetime

# Global results list
RESULTS = []
RESULTS_FILE = None

def init_results_file():
    global RESULTS_FILE
    results_dir = "/workspace/PyTorchSim/togsim_results"
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

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4, m=None, n=None, k=None):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        log_result("-" * len(message))
        log_result(message)
        log_result("-" * len(message))

        # Print matrix dimensions and result statistics
        if m is not None and n is not None and k is not None:
            log_result(f"Matrix: {m}×{k}×{n}")
            log_result(f"Output sum: {out.sum().item():.4f}")
            log_result(f"Output min: {out.min().item():.4f}, max: {out.max().item():.4f}")

            # Print tensor output
            log_result(f"custom out: {out.cpu()}")
            log_result(f"cpu out: {cpu_out}")

        # Print performance metrics summary
        log_result("")
        log_result("SIMULATION RESULTS:")

        # Try to extract performance metrics from TOGSim log
        try:
            log_dir = "/workspace/PyTorchSim/togsim_results"
            if os.path.exists(log_dir):
                # Find latest log file
                log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')])
                if log_files:
                    latest_log = os.path.join(log_dir, log_files[-1])
                    log_result(f"Log file: {os.path.basename(latest_log)}")
                    with open(latest_log, 'r') as f:
                        content = f.read()
                        # Extract metrics
                        cycles_match = re.search(r'Total cycle: (\d+)', content)
                        systolic_match = re.search(r'Systolic Array Utilization\(%\) ([\d.]+)', content)
                        vector_match = re.search(r'Vector Unit Utilization\(%\) ([\d.]+)', content)
                        bw_match = re.search(r'DRAM: AVG BW Util ([\d.]+)%', content)

                        if cycles_match:
                            log_result(f"Total cycles: {cycles_match.group(1):>8}")
                        if systolic_match:
                            log_result(f"Systolic [0]: {systolic_match.group(1):>8}%")
                        if vector_match:
                            log_result(f"Vector Unit: {vector_match.group(1):>8}%")
                        if bw_match:
                            log_result(f"DRAM BW: {bw_match.group(1):>8}%")
        except Exception as e:
            log_result(f"(Performance metrics unavailable)")

        log_result("")

    else:
        message = f"|{name} Test Failed|"
        log_result("-" * len(message))
        log_result(message)
        log_result("-" * len(message))
        log_result("custom out: " + str(out.cpu()))
        log_result("cpu out: " + str(cpu_out))
        log_result(f"Results saved to: {RESULTS_FILE}")
        exit(1)

def test_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(x1, w1)
    y = custom_matmul(x2, w2)
    test_result("Matmul Forward", res, y, m=input_size, n=output_size, k=hidden_size)

def test_addmm(device, input_size=128, hidden_size=128, output_size=128, bias_rank=1):
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
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)
    y = custom_matmul(b2, x2, w2)
    test_result("Addmm Forward", res, y, m=input_size, n=output_size, k=hidden_size)

def test_addmm2(device, input_size=128, hidden_size=128, output_size=128):
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
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)
    y = custom_matmul(b2, x2, w2)
    test_result("Addmm2 Forward", res, y, m=input_size, n=output_size, k=hidden_size)

def test_linear(device, input_size=128, hidden_size=128, output_size=128):
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
    opt_fn = torch.compile(dynamic=False)(custom_linear)
    res = opt_fn(x1, w1, b1)
    y = custom_linear(x2, w2, b2)
    test_result("Linear Forward", res, y, m=input_size, n=output_size, k=hidden_size)

if __name__ == "__main__":
    init_results_file()
    device = torch.device("npu:0")
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
