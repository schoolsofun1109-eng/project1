#!/bin/bash
# PyTorchSim 테스트 실행
# 설정: 이 스크립트를 실행하면 기본값으로 테스트를 수행합니다
# 테스트 함수나 파라미터를 변경하려면 아래의 변수를 수정하세요

FUNC=${FUNC:-matmul} #matmul, BMM
BATCH_SIZE=${BATCH_SIZE:-2}
M=${M:-256}
N=${N:-128}
K=${K:-128}
CONFIG=${CONFIG:-/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_imem_wmem_omem_64kb.yml}

echo "=========================================="
echo "PyTorchSim Test"
echo "=========================================="
echo "Function: $FUNC"
echo "Matrix size: $M × $N × $K"
echo "Config: $(basename $CONFIG)"
echo ""

docker run --rm --runtime=runc \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/PyTorchSimFrontend:/workspace/PyTorchSim/PyTorchSimFrontend \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/configs:/workspace/PyTorchSim/configs \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/togsim_results:/workspace/PyTorchSim/togsim_results \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/gem5_script:/workspace/PyTorchSim/gem5_script \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/tests:/workspace/PyTorchSim/tests \
  ghcr.io/psal-postech/torchsim-ci:v1.1.0 \
  bash -c "
cd /workspace/PyTorchSim && rm -rf outputs && \
PYTHONUNBUFFERED=1 TOGSIM_CONFIG=$CONFIG python -u -c \"
import torch
import importlib
import inspect

# Normalize function name
func_input = '$FUNC'
if func_input.startswith('test_'):
    func_input = func_input[5:]
func_name = f'test_{func_input}'

# Find function in test modules
module = None
for mod_name in ['test_bmm', 'test_matmul']:
    try:
        module = importlib.import_module(f'tests.{mod_name}')
        if hasattr(module, func_name):
            break
    except ImportError:
        pass

if module is None or not hasattr(module, func_name):
    print(f'Error: function {func_name} not found in test_bmm or test_matmul')
    exit(1)

func = getattr(module, func_name)
device = torch.device('npu:0')

# Inspect function signature and call appropriately
sig = inspect.signature(func)
params = list(sig.parameters.keys())

# Call function with correct arguments based on signature
if 'batch_size' in params:
    # test_bmm style: (device, batch_size, m, n, k)
    func(device, $BATCH_SIZE, $M, $N, $K)
else:
    # test_matmul style: (device, input_size, hidden_size, output_size)
    func(device, $M, $N, $K)
\" 2>&1 | tee /tmp/test_output.log; \
\
echo '' && \
echo '======================================' && \
echo 'SIMULATION RESULTS' && \
echo '======================================' && \
LATEST_LOG=\$(ls -t togsim_results/*.log 2>/dev/null | head -1) && \
if [ -n \"\$LATEST_LOG\" ]; then \
  echo \"Log file: \$LATEST_LOG\" && \
  CYCLES=\$(grep 'Total execution cycles:' \$LATEST_LOG | tail -1 | grep -o '[0-9]*\$'); \
  UTIL0=\$(grep 'Systolic array \[0\] utilization' \$LATEST_LOG | tail -1 | grep -oE 'utilization\(%\): [0-9.]+' | grep -oE '[0-9.]+'); \
  UTIL1=\$(grep 'Systolic array \[1\] utilization' \$LATEST_LOG | tail -1 | grep -oE 'utilization\(%\): [0-9.]+' | grep -oE '[0-9.]+'); \
  BW=\$(grep 'channels 0\.\.15 combined' \$LATEST_LOG | tail -1 | grep -oE '[0-9.]+ GB/s' | head -1); \
  echo \"Total cycles:    \$CYCLES\" && \
  echo \"Systolic [0]:    \$UTIL0%\" && \
  echo \"Systolic [1]:    \$UTIL1%\" && \
  echo \"DRAM BW:         \$BW\"; \
fi && \
echo ''
" 2>&1 | tail -100

