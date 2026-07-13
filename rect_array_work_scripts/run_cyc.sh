#!/bin/bash
M=${M:-128}; N=${N:-128}; K=${K:-128}
CONFIG=${CONFIG:?}; MLIROPT=${MLIROPT:-}
MNT=""; [ -n "$MLIROPT" ] && MNT="-v $MLIROPT:/riscv-llvm/bin/mlir-opt"
docker run --rm --runtime=runc \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/PyTorchSimFrontend:/workspace/PyTorchSim/PyTorchSimFrontend \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/configs:/workspace/PyTorchSim/configs \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/gem5_script:/workspace/PyTorchSim/gem5_script \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/tests:/workspace/PyTorchSim/tests \
  -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/Simulator:/workspace/PyTorchSim/Simulator \
  $MNT ghcr.io/psal-postech/torchsim-ci:v1.1.0 \
  bash -c "cd /workspace/PyTorchSim && rm -rf outputs togsim_results && mkdir -p togsim_results && \
    PYTHONUNBUFFERED=1 TOGSIM_CONFIG=$CONFIG python -u -c \"
import torch, importlib
importlib.import_module('tests.test_matmul').test_matmul(torch.device('npu:0'), $M,$N,$K)
\" >/tmp/o.log 2>&1; \
    L=\$(ls -t togsim_results/*.log 2>/dev/null | head -1); \
    echo -n 'RECT-PE: '; grep -h 'RECT-PE' /tmp/o.log \$L 2>/dev/null | tail -1; \
    echo -n 'Cycles: '; grep -hoE 'Total execution cycles: [0-9]+|Cycles: [0-9]+' /tmp/o.log \$L 2>/dev/null | grep -oE '[0-9]+' | tail -1; \
    echo -n 'PassFail: '; grep -hoE 'Test Passed|Test Failed' /tmp/o.log 2>/dev/null | tail -1; \
    grep -h '3D-PE' /tmp/o.log 2>/dev/null | tail -1" 2>&1 | grep -vE "CUDA|NVIDIA|====|License|governed|pulling|https|Use the|copy of|Container image|By pulling|A copy|^$"
