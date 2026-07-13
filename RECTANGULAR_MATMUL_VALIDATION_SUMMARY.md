# Rectangular Matmul Tile Stride Fix Validation Report

## Executive Summary

**Status: ❌ VALIDATION FAILED**

The tile stride fix (commit 27ede7a) does **NOT** resolve rectangular matmul failures. Only square matrices (M=N) pass.

- **Tests Passed:** 3/10 (30%)
- **Tests Failed:** 7/10 (70%)
- **Expected:** 10/10 (100%)

---

## Quick Facts

| Metric | Value |
|--------|-------|
| Fix Commit | 27ede7a |
| Test Suite | 10 Docker-based tests |
| Execution Time | ~40 minutes |
| Platform | ghcr.io/psal-postech/torchsim-ci:v1.1.0 |
| Success Rate | 30% (Need 100%) |

---

## Test Results

### Passing Tests (Square Matrices)
```
✓ Test 6:  test_matmul(32×32×32)       max_diff = 2.2e-05
✓ Test 7:  test_matmul(128×128×128)    max_diff = 1.7e-06  
✓ Test 8:  test_matmul(256×256×256)    max_diff = 2.3e-05
```

### Failing Tests (Rectangular Matrices)
```
✗ Test 1:  test_matmul(128×256×256)    max_diff = 78.04
✗ Test 2:  test_matmul(128×63×56)      max_diff = 19.89
✗ Test 3:  test_addmm(128×256×512)     max_diff = 69.44
✗ Test 4:  test_addmm(128×256×512)     max_diff = 69.44
✗ Test 5:  test_addmm(129×61×56)       max_diff = 20.22
✗ Test 9:  test_addmm2(129×61×56)      max_diff = 20.22
✗ Test 10: test_addmm(516×244×224)     max_diff = 60.61
```

---

## Critical Finding

**Tests pass ONLY when M = N (square matrices)**
**Tests fail for ALL M ≠ N (rectangular matrices)**

This indicates:
1. The tile stride fix is mathematically correct
2. But operationally incomplete
3. The root cause is NOT the stride formula
4. The root cause is in split-iteration DMA or SRAM allocation

---

## The Fix Explained

### What Changed (Commit 27ede7a)

```python
# mlir_gemm_template.py

# BEFORE (incorrect):
X_tile_stride = [TILE_M, 1]      # Wrong: X has TILE_K columns, not TILE_M
W_tile_stride = [TILE_K, 1]      # Wrong: W has TILE_N columns, not TILE_K
Y_tile_stride = [TILE_M, 1]      # Wrong: Y has TILE_N columns, not TILE_M

# AFTER (correct):
X_tile_stride = [TILE_K, 1]      # Right: X[TILE_M, TILE_K] → stride TILE_K
W_tile_stride = [TILE_N, 1]      # Right: W[TILE_K, TILE_N] → stride TILE_N
Y_tile_stride = [TILE_N, 1]      # Right: Y[TILE_M, TILE_N] → stride TILE_N
```

### Why It's Correct (Mathematically)

For row-major layout, stride = [columns, 1]:
- X has shape [TILE_M, TILE_K] → stride [TILE_K, 1] ✓
- W has shape [TILE_K, TILE_N] → stride [TILE_N, 1] ✓
- Y has shape [TILE_M, TILE_N] → stride [TILE_N, 1] ✓

### Why It Passes for Squares (By Accident)

```
When M = N:
  TILE_M = TILE_N (tiles are square)
  Original [TILE_M, 1] and fixed [TILE_K, 1] both work
  Fix passes for square case coincidentally
```

### Why It Fails for Rectangles (The Problem)

```
When M ≠ N:
  TILE_M ≠ TILE_N (tiles are rectangular)
  Stride definition alone is insufficient
  Real issue in split-iteration DMA or SRAM allocation
```

---

## Performance Degradation

### Systolic Array Utilization

**Square Cases (PASS):**
- 32×32×32: 100% utilization
- 128×128×128: 100% utilization
- 256×256×256: 100% utilization

**Rectangular Cases (FAIL):**
- 128×256×256: 18.6% utilization ⚠️ (should be 100%)
- 128×63×56: 9.3% utilization ⚠️ (should be 100%)
- 128×256×512: 6.3% utilization ⚠️ (should be 100%)

Low utilization confirms tiles are not being efficiently used.

---

## Failure Pattern Analysis

### Test 1: 128×256×256
```
TILE_M=64, TILE_N=32, TILE_K=64 (rectangular!)
- First tile: OK
- Second tile: Massive errors (max_diff=78.0)
→ Suggests: N-split offset calculation error
```

### Test 2: 128×63×56
```
First rows: Correct match with CPU
Last row: All zeros
→ Suggests: Buffer allocation overflow or N-split tile failure
```

### Test 9: 129×61×56
```
Partial tiles: Correct values
Middle rows: All zeros
→ Suggests: Partial SRAM allocation failure
```

---

## Root Cause Hypothesis

The tile stride fix is necessary but not sufficient. The real root cause is likely one of:

1. **Split-Iteration DMA Address Calculation** (HIGHEST PROBABILITY)
   - File: `mlir_gemm_template.py` lines 170-180
   - Problem: DMA indices use DRAM stride, not SRAM stride
   - For rectangular tiles, split-iteration offsets are wrong

2. **SRAM Buffer Allocation for Separated IMEM/WMEM/OMEM**
   - File: `mlir_common.py` or `tile_allocation.py`
   - Problem: Buffer sizing assumes square tiles
   - For rectangular tiles, allocation fails

3. **Vector Lane Distribution for Non-Square Tiles**
   - File: `mlir_common.py` - MLIRMultiDimTile class
   - Problem: Vector lane mapping assumes TILE_M = TILE_N
   - For rectangular tiles, lane distribution incorrect

4. **N-Split Tile Indexing**
   - File: `mlir_gemm_template.py` - N-split loop handling
   - Problem: Tile offset calculations assume uniform sizing
   - For rectangular tiles with M≠N, indexing breaks

---

## Recommendation

### DO NOT Revert Commit 27ede7a
- The stride definition IS correct
- It helps (squares pass)
- But it must be paired with additional fixes

### DO Investigate These Areas
1. Split-iteration DMA address calculation in mlir_gemm_template.py
2. SRAM buffer allocation for non-square tiles
3. Vector lane distribution for rectangular tiles
4. N-split tile indexing for M≠N cases

### DO Test These Hypotheses
1. Run single-split case (M-split=1, N-split=1) with rectangular matrix
2. If single-split passes, problem is in split-iteration code
3. If single-split fails, problem is in core DMA code

### DO Add Debug Instrumentation
1. Log DMA offset calculations
2. Verify buffer allocation sizes
3. Check N-split tile indexing
4. Profile systolic array usage per tile

---

## Files for Reference

- **Main Report:** `/home/sslunder63/project/VQ_NPU_Simulator/FINAL_VALIDATION_REPORT.txt`
- **Root Cause Analysis:** `/home/sslunder63/project/VQ_NPU_Simulator/ROOT_CAUSE_ANALYSIS.md`
- **Test Data:** `/home/sslunder63/project/VQ_NPU_Simulator/COMPREHENSIVE_TEST_DATA.csv`
- **Validation Summary:** `/home/sslunder63/project/VQ_NPU_Simulator/VALIDATION_SUMMARY.txt`
- **Individual Test Logs:** `/tmp/rectangular_test_results/test_*.log`

---

## Conclusion

The tile stride fix (commit 27ede7a) is:
- ✅ **Mathematically correct**
- ❌ **Operationally incomplete**

The fix resolves stride formula errors but does not resolve the root cause of rectangular matmul failures. Additional changes are needed in split-iteration DMA addressing or SRAM allocation.

**Success Rate:** 3/10 (30%) → **Must reach 10/10 (100%)**
