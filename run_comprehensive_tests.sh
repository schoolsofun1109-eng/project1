#!/bin/bash
# Comprehensive rectangular matmul test suite
# Tests all 10 cases: 5 previously failing + 5 previously passing

set -e

TEST_CONFIG="/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_imem_wmem_omem_64kb.yml"
DOCKER_IMAGE="ghcr.io/psal-postech/torchsim-ci:v1.1.0"

# Array of test cases: (name, M, N, K, was_passing)
declare -a TESTS=(
    "128x256x256|128|256|256|0"      # Test 1: Was diff=78
    "256x512x256|256|512|256|0"      # Test 2: Was diff=143
    "512x256x256|512|256|256|0"      # Test 3: Was diff=59
    "128x128x1|128|128|1|0"          # Test 4: Was FAIL
    "3x256x128|3|256|128|0"          # Test 5: Was TIMEOUT
    "64x128x64|64|128|64|1"          # Test 6: Previously passing
    "64x256x128|64|256|128|1"        # Test 7: Previously passing
    "256x128x64|256|128|64|1"        # Test 8: Previously passing
    "256x64x128|256|64|128|1"        # Test 9: Previously passing
    "1x128x128|1|128|128|1"          # Test 10: Edge case
)

# Result tracking
declare -A RESULTS
declare -A ERRORS
PASS_COUNT=0
FAIL_COUNT=0

echo "================================================================================"
echo "COMPREHENSIVE RECTANGULAR MATMUL TEST SUITE"
echo "================================================================================"
echo "Test Config: $TEST_CONFIG"
echo "Docker Image: $DOCKER_IMAGE"
echo ""
echo "Test Matrix:"
echo "  1-5: Previously FAILING (now expect PASS)"
echo "  6-10: Previously PASSING (regression check)"
echo ""
echo "================================================================================"
echo ""

# Run each test
for i in "${!TESTS[@]}"; do
    IFS='|' read -r NAME M N K WAS_PASSING <<< "${TESTS[$i]}"
    TEST_NUM=$((i + 1))

    echo "================================================================================
[TEST $TEST_NUM/$((${#TESTS[@]}))] $NAME (${M}×${N}×${K})"
    echo "================================================================================"

    if [ "$WAS_PASSING" = "1" ]; then
        echo "[REGRESSION CHECK] Previously passing"
    else
        echo "[PRIORITY FIX] Previously failing"
    fi
    echo ""

    # Run Docker test
    OUTPUT=$(docker run --rm --runtime=runc \
      -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/PyTorchSimFrontend:/workspace/PyTorchSim/PyTorchSimFrontend \
      -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/configs:/workspace/PyTorchSim/configs \
      -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/togsim_results:/workspace/PyTorchSim/togsim_results \
      -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/gem5_script:/workspace/PyTorchSim/gem5_script \
      -v /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim/tests:/workspace/PyTorchSim/tests \
      "$DOCKER_IMAGE" \
      bash -c "
cd /workspace/PyTorchSim && rm -rf outputs && \
PYTHONUNBUFFERED=1 TOGSIM_CONFIG=$TEST_CONFIG timeout 120 python -u -c \"
import sys
import torch

# Import test module
from tests.test_matmul import test_matmul

try:
    device = torch.device('npu:0')
    test_matmul(device, $M, $N, $K)
    print('\\n[TEST_RESULT] SUCCESS')
    sys.exit(0)
except Exception as e:
    print(f'\\n[TEST_RESULT] FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
\" 2>&1
" 2>&1) || true

    # Parse results
    if echo "$OUTPUT" | grep -q "Test Passed"; then
        RESULTS[$TEST_NUM]="PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "✓ PASSED"

        # Extract max diff
        MAX_DIFF=$(echo "$OUTPUT" | grep "Max absolute difference:" | tail -1 | grep -oE "[0-9.e+-]+" | head -1)
        if [ -n "$MAX_DIFF" ]; then
            echo "  Max absolute diff: $MAX_DIFF"
        fi
    else
        RESULTS[$TEST_NUM]="FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ FAILED"
        ERRORS[$TEST_NUM]="$OUTPUT"
    fi
    echo ""
done

# Summary report
echo "================================================================================"
echo "COMPREHENSIVE TEST RESULTS SUMMARY"
echo "================================================================================"
echo ""
echo "Pass/Fail Count: $PASS_COUNT/$((${#TESTS[@]})) PASS"
if [ $FAIL_COUNT -gt 0 ]; then
    echo "Failed Tests: $FAIL_COUNT"
fi
echo ""

# Detailed results table
echo "Test | Case            | M    | N    | K    | Status | Category"
echo "-----|-----------------|------|------|------|--------|------------"

for i in "${!TESTS[@]}"; do
    IFS='|' read -r NAME M N K WAS_PASSING <<< "${TESTS[$i]}"
    TEST_NUM=$((i + 1))
    STATUS=${RESULTS[$TEST_NUM]:-"ERROR"}
    if [ "$WAS_PASSING" = "1" ]; then
        CATEGORY="Regression"
    else
        CATEGORY="Priority"
    fi
    printf "%4d | %-15s | %4d | %4d | %4d | %6s | %s\n" "$TEST_NUM" "$NAME" "$M" "$N" "$K" "$STATUS" "$CATEGORY"
done

echo ""
echo "================================================================================"
echo "PASS RATE: $PASS_COUNT/$((${#TESTS[@]})) ($(( PASS_COUNT * 100 / ${#TESTS[@]} ))%)"
echo "================================================================================"
echo ""

if [ $PASS_COUNT -eq ${#TESTS[@]} ]; then
    echo "✅ SUCCESS: ALL 10/10 RECTANGULAR MATMUL TESTS PASSING"
    echo "The SRAM tile_stride fix has successfully resolved all M≠N matmul failures."
    exit 0
else
    echo "❌ INCOMPLETE: $FAIL_COUNT test(s) still failing"
    if [ $FAIL_COUNT -gt 0 ]; then
        echo ""
        echo "Failed test details:"
        for i in "${!TESTS[@]}"; do
            TEST_NUM=$((i + 1))
            if [ "${RESULTS[$TEST_NUM]}" = "FAIL" ]; then
                IFS='|' read -r NAME M N K WAS_PASSING <<< "${TESTS[$i]}"
                echo ""
                echo "--- Test $TEST_NUM: $NAME ---"
                echo "${ERRORS[$TEST_NUM]}" | tail -50
            fi
        done
    fi
    exit 1
fi
