#!/bin/bash
# Args via env: M N K CONFIG MLIROPT(optional host path to bind-mount) VLANE_HEIGHT(optional)
M=${M:-128}; N=${N:-128}; K=${K:-128}
CONFIG=${CONFIG:-/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_imem_wmem_omem_64kb.yml}
FUNC=${FUNC:-matmul}
MLIROPT_MOUNT=""
if [ -n "$MLIROPT" ]; then MLIROPT_MOUNT="-v $MLIROPT:/riscv-llvm/bin/mlir-opt"; fi
docker run --rm --runtime=runc \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/PyTorchSimFrontend:/workspace/PyTorchSim/PyTorchSimFrontend \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/configs:/workspace/PyTorchSim/configs \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/gem5_script:/workspace/PyTorchSim/gem5_script \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/tests:/workspace/PyTorchSim/tests \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/Simulator:/workspace/PyTorchSim/Simulator \
  $MLIROPT_MOUNT \
  ghcr.io/psal-postech/torchsim-ci:v1.1.0 \
  bash -c "cd /workspace/PyTorchSim && rm -rf outputs togsim_results && mkdir -p togsim_results && \
    PYTHONUNBUFFERED=1 TOGSIM_CONFIG=$CONFIG python -u -c \"
import torch, importlib
m = importlib.import_module('tests.test_matmul')
torch_dev = torch.device('npu:0')
m.test_matmul(torch_dev, $M, $N, $K)
\" 2>&1 | tail -40; \
    L=\$(ls -t togsim_results/*.log 2>/dev/null | head -1); \
    echo \"=== CYCLES ===\"; grep 'Total execution cycles:' \$L | tail -1; \
    grep 'RECT-PE' \$L 2>/dev/null | tail -2 " 2>&1 | grep -vE "CUDA|NVIDIA|====|License|governed|pulling|https|Use the|copy of|^$|Container image|By pulling|A copy"
