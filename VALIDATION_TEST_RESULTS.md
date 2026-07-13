# Rectangular Matmul Validation Test Results

**Test Date**: 2026-07-03  
**Docker Image**: ghcr.io/psal-postech/torchsim-ci:v1.1.0  
**Test Config**: systolic_ws_128x128_c1_imem_wmem_omem_64kb.yml

---

## Executive Summary

**Result: FAILED - 0/10 passing**

The SRAM tile_stride fix (commit a003c2f) and related MLIR fixes (commits 2ce04ce, 95019b1) did NOT successfully resolve the rectangular matmul failures. Instead, ALL tests are failing, including previously passing cases, indicating a critical regression.

---

## Test Results Summary

| Test # | Case            | M   | N   | K   | Category   | Status | Max Diff  | Issue                          |
|--------|-----------------|-----|-----|-----|------------|--------|-----------|--------------------------------|
| 1      | 128×256×256    | 128 | 256 | 256 | Priority   | FAIL   | 82.88     | Zeros in bottom-right region   |
| 2      | 256×512×256    | 256 | 512 | 256 | Priority   | FAIL   | 137.86    | Zeros in bottom-right region   |
| 3      | 512×256×256    | 512 | 256 | 256 | Priority   | FAIL   | 88.30     | Zeros in bottom-right region   |
| 4      | 128×128×1     | 128 | 128 | 1   | Priority   | FAIL   | N/A       | Incomplete output              |
| 5      | 3×256×128     | 3   | 256 | 128 | Priority   | FAIL   | N/A       | AssertionError in MLIRCodeCache|
| 6      | 64×128×64     | 64  | 128 | 64  | Regression | FAIL   | 59.07     | Zeros in bottom-right region   |
| 7      | 64×256×128    | 64  | 256 | 128 | Regression | FAIL   | 80.00     | Zeros in bottom-right region   |
| 8      | 256×128×64    | 256 | 128 | 64  | Regression | FAIL   | 56.40     | Zeros in bottom-right region   |
| 9      | 256×64×128    | 256 | 64  | 128 | Regression | FAIL   | 41.60     | Zeros in bottom-right region   |
| 10     | 1×128×128     | 1   | 128 | 128 | Regression | FAIL   | 29.29     | Incomplete output              |

**Pass Rate: 0% (0/10 PASS)**

---

## Critical Findings

### 1. Regression in All Tests
- Even previously passing cases (tests 6-10) are now failing
- This indicates the fixes introduced a critical bug rather than resolving the issue

### 2. Consistent Failure Pattern: Zeros in Output
Most tests show a distinctive pattern:
```
Custom out: tensor([[ real_values, real_values, ...  ],
                    [ real_values, real_values, ...  ],
                    ...
                    [ 0.0000, 0.0000, 0.0000, ... ],
                    [ 0.0000, 0.0000, 0.0000, ... ]])
```

The bottom-right portion of the output matrix contains zeros instead of computed values. This suggests:
- **Buffer overflow/underflow** in SRAM allocation
- **Incorrect loop bounds** in the kernel
- **Tile boundary issue** affecting the last rows/columns

### 3. Test 128×128×128 (Square Matrix) Failing
Even the square 128×128×128 matrix fails with max_diff = 57.6, which is unacceptable:
```
Max absolute difference: 5.760365e+01
Max relative difference: 4379.9927
```

This should be one of the easiest cases to handle and the failure indicates a fundamental issue with the tile_stride change.

### 4. Test 3×256×128 Crashes  
AssertionError at MLIRCodeCache.load() - suggests code generation failure for non-power-of-2 M dimension.

---

## Code Analysis

### Commits Applied (In Order)
1. **2ce04ce**: Fix tile descriptor and DRAM stride handling
   - Fixed W_tile_desc using X_tile_size (should use W_tile_size)
   - Removed conditional on W_stride and Y_stride
   
2. **95019b1**: Fix MLIR variable linking in split iteration DMA
   - Fixed symbol extraction in affine.apply operations
   
3. **a003c2f**: Fix SRAM tile_stride (SUSPECTED CULPRIT)
   - Changed X_tile_stride: `[1, TILE_M]` → `[TILE_M, 1]`
   - Changed W_tile_stride: `[1, TILE_K]` → `[TILE_K, 1]`
   - Changed Y_tile_stride: `[1, TILE_M]` → `[TILE_M, 1]`

The tile_stride change assumes row-major layout in SRAM, but this may be incompatible with how the SRAM is actually being accessed in the compiled kernel.

---

## Root Cause Analysis

### Hypothesis 1: Incorrect Stride Change
The tile_stride represents how elements are laid out in SRAM. Changing from column-major `[1, TILE_M]` to row-major `[TILE_M, 1]` may have broken assumptions in:
- DMA operations (allocate_sram)
- Vector lane layout calculations  
- Affine loop index calculations

### Hypothesis 2: Incomplete Fix Chain
The three commits may not be sufficient or may conflict:
- The W_tile_desc fix (2ce04ce) changes how tiles are described
- The MLIR linking fix (95019b1) changes DMA generation
- The stride fix (a003c2f) changes memory layout
- These may not be compatible together

### Hypothesis 3: Configuration Mismatch
The config uses IMEM/WMEM/OMEM separation (64KB total):
- IMEM: 32KB @ 0xd0000000
- WMEM: 16KB @ 0xd0008000  
- OMEM: 16KB @ 0xd000c000

The stride changes might not account for per-memory-role constraints properly.

---

## Conclusions

The current fixes are **NOT VALIDATED**. The test results indicate:

1. **The fixes made the problem worse**, not better
2. **All tests are regressing**, including previously passing ones
3. **The output pattern (zeros in bottom-right) suggests buffer management issues**
4. **A fundamental aspect of SRAM layout was likely misunderstood**

### Recommended Next Steps

1. **Revert the tile_stride fix (a003c2f)** and test with just commits 2ce04ce + 95019b1
2. **Investigate allocate_sram()** to understand how tile_stride is used
3. **Review DRAM stride vs SRAM stride** - they may serve different purposes
4. **Check if the issue is in vector lane calculations** rather than basic stride
5. **Test incrementally** - apply each commit separately and measure impact

### Action Required

Do NOT merge these changes. The comprehensive validation clearly shows these fixes have introduced critical regressions.
