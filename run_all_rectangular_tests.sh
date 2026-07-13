#!/bin/bash

# Run all 10 rectangular matmul tests with the tile stride fix
# Tests 1-5: Rectangular cases
# Tests 6-10: Previously passing cases (should not regress)

CONFIG=${CONFIG:-/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_imem_wmem_omem_64kb.yml}

# Create temporary directory for results
RESULTS_DIR="/tmp/rectangular_test_results"
mkdir -p "$RESULTS_DIR"

# Function to run a single test
run_test() {
    local test_num=$1
    local test_func=$2
    local m=$3
    local n=$4
    local k=$5
    local test_name="$test_func(${m}x${n}x${k})"

    echo ""
    echo "=========================================="
    echo "Test $test_num: $test_name"
    echo "=========================================="

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
import sys
sys.path.insert(0, '.')
from tests.test_matmul import test_matmul, test_addmm, test_addmm2

# Call the appropriate test
test_name = '$test_func'
m, n, k = $m, $n, $k

device = torch.device('npu:0')

try:
    if test_name == 'test_matmul':
        test_matmul(device, m, n, k)
        print(f'\n✓ Test $test_num PASSED')
    elif test_name == 'test_addmm':
        test_addmm(device, m, n, k)
        print(f'\n✓ Test $test_num PASSED')
    elif test_name == 'test_addmm2':
        test_addmm2(device, m, n, k)
        print(f'\n✓ Test $test_num PASSED')
except Exception as e:
    print(f'\n✗ Test $test_num FAILED: {e}')
    sys.exit(1)
\" 2>&1
" > "$RESULTS_DIR/test_${test_num}.log" 2>&1

    # Check result
    if grep -q "PASSED" "$RESULTS_DIR/test_${test_num}.log"; then
        echo "✓ Test $test_num PASSED"
        return 0
    else
        echo "✗ Test $test_num FAILED"
        tail -50 "$RESULTS_DIR/test_${test_num}.log"
        return 1
    fi
}

# Initialize results tracking
PASSED=0
FAILED=0
FAILED_TESTS=""

echo "=========================================="
echo "Running All 10 Rectangular Matmul Tests"
echo "=========================================="
echo "Config: $CONFIG"
echo ""

# Test 1: Rectangular 128×256×256
if run_test 1 "test_matmul" 128 256 256; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 1"; fi

# Test 2: Rectangular 128×63×56
if run_test 2 "test_matmul" 128 63 56; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 2"; fi

# Test 3: Rectangular with addmm 128×256×512
if run_test 3 "test_addmm" 128 256 512; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 3"; fi

# Test 4: Rectangular with addmm (bias_rank=2) 128×256×512
if run_test 4 "test_addmm" 128 256 512; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 4"; fi

# Test 5: Rectangular with addmm 129×61×56
if run_test 5 "test_addmm" 129 61 56; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 5"; fi

# Test 6: Non-rectangular (previously passing) 32×32×32
if run_test 6 "test_matmul" 32 32 32; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 6"; fi

# Test 7: Non-rectangular (previously passing) 128×128×128
if run_test 7 "test_matmul" 128 128 128; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 7"; fi

# Test 8: Non-rectangular (previously passing) 256×256×256
if run_test 8 "test_matmul" 256 256 256; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 8"; fi

# Test 9: addmm2 129×61×56
if run_test 9 "test_addmm2" 129 61 56; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 9"; fi

# Test 10: addmm 129×61×56 scaled
if run_test 10 "test_addmm" 516 244 224; then ((PASSED++)); else ((FAILED++)); FAILED_TESTS="$FAILED_TESTS 10"; fi

# Summary
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Passed: $PASSED/10"
echo "Failed: $FAILED/10"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: 10/10 RECTANGULAR MATMUL NOW PASSING"
    exit 0
else
    echo ""
    echo "❌ FAILURE: Some tests failed"
    echo "Failed tests: $FAILED_TESTS"
    exit 1
fi
