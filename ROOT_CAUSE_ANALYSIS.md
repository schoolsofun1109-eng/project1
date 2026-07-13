# Root Cause Analysis: Why Tile Stride Fix Fails for Rectangular Matmul

## Summary
The tile stride fix (commit 27ede7a) is **mathematically correct** but **operationally incomplete**. It works for square matrices (M=N) by accident, but fails for all rectangular cases (M≠N).

## The Fix (Correct but Incomplete)

Commit 27ede7a changes:
```python
# mlir_gemm_template.py lines 150-177
X_tile_stride = [TILE_K, 1]       # was [TILE_M, 1]  
W_tile_stride = [TILE_N, 1]       # was [TILE_K, 1]
Y_tile_stride = [TILE_N, 1]       # was [TILE_M, 1]
```

### Why This Fix is Correct (Mathematically)

For row-major matrix layout, stride = [width, 1]:
- X[TILE_M, TILE_K]: next row requires skipping TILE_K elements → [TILE_K, 1] ✓
- W[TILE_K, TILE_N]: next row requires skipping TILE_N elements → [TILE_N, 1] ✓  
- Y[TILE_M, TILE_N]: next row requires skipping TILE_N elements → [TILE_N, 1] ✓

### Why It Passes for Square Matrices (Accidental)

When M = N (square case):
- TILE_M = TILE_N (tiles are also square)
- Original [TILE_M, 1] and corrected [TILE_K, 1] happen to work because...
  - For X: TILE_M and TILE_K are both ≤ systolic_array_size
  - For square GEMM: K-dimension iterations match M-dimension tiles
  - Vector lane distribution works for equal tile dimensions

### Why It Fails for Rectangular Matrices (The Real Problem)

When M ≠ N (rectangular case):
- TILE_M ≠ TILE_N
- The stride fix alone is insufficient

## The Real Problem: Not Stride, But Usage

Test failure analysis shows:
1. **Partial tile corruption**: Some tiles are correct, others are all zeros
2. **Systolic utilization**: Only 18.6% for rectangular vs 100% for square
3. **Tile boundary issues**: Last row/column of some tiles missing

This pattern indicates: **SRAM allocation failure, not stride formula failure**

## Suspected Root Causes (In Order of Likelihood)

### 1. **DMA Address Calculation for Split Iterations** (HIGHEST PROBABILITY)

File: `mlir_gemm_template.py` lines 170-180

Current code:
```python
X_idx = [sympy.Symbol("index0") * X_stride[0], sympy.Symbol("index2") * X_stride[1]]
W_idx = [sympy.Symbol("index2") * W_stride[0], sympy.Symbol("index1") * W_stride[1]]
Y_idx = [sympy.Symbol("index0") * Y_stride[0], sympy.Symbol("index1") * Y_stride[1]]
```

Problem:
- These indices use DRAM stride (full matrix width), not tile stride
- For rectangular matrices, when N ≠ M, tile layout in SRAM differs from DRAM
- N-split and M-split iterations need different offset calculations

### 2. **SRAM Allocation for Separated IMEM/WMEM/OMEM**

File: `mlir_common.py` or `tile_allocation.py`

Problem:
- Separate memory sections assume certain tile dimensions
- When TILE_M ≠ TILE_N, buffer sizing calculations fail
- Evidence: Low systolic utilization (18.6% vs 100%)

### 3. **Vector Lane Distribution for Non-Square Tiles**

File: `mlir_common.py` - `MLIRMultiDimTile` class

Problem:
- Vector lane distribution assumes square tiles
- For TILE_M=64, TILE_N=32 case, vector lane mapping might be incorrect
- This would explain why partial tiles work and others fail

### 4. **N-Split Tile Indexing**

File: `mlir_gemm_template.py` - N-split loop handling

Problem:
- When tiles are split across multiple N values
- Current code might calculate offsets assuming uniform tile sizes
- Rectangular tiles require different offset calculations

## Test Evidence Supporting Each Cause

### Test 1: 128×256×256 (TILE_M=64, TILE_N=32, TILE_K=64)
- First output matches CPU
- Second output column has massive errors
- **Suggests**: N-split offset calculation is wrong

### Test 2: 128×63×56 (Highly rectangular)
- First rows correct, last row all zeros
- **Suggests**: Buffer allocation ran out of space or offset overflow

### Test 9: 129×61×56 (After addmm2)
- Partial tile matches, zeros in middle rows
- **Suggests**: Vector lane distribution or split-iteration addressing

## Fix Strategy (Priority Order)

### Priority 1: Verify DMA Index Calculation
```python
# Check mlir_gemm_template.py line 172
W_idx = [sympy.Symbol("index2") * W_stride[0], ...]
#      Should W_stride here use SRAM stride [TILE_N, 1]
#      or DRAM stride (full N)?
```

Verify in split iteration loop:
```python
for i_n in range(0, N, TILE_N):  # N-split loop
    # W_idx calculation must account for:
    # - DRAM offset from N position
    # - SRAM position for this tile
    # - Current stride values
```

### Priority 2: Check SRAM Buffer Sizing
```python
# mlir_common.py - MLIRMultiDimTile.set_tile_size_stride()
# Verify that buffer allocation accounts for TILE_M × TILE_N
# Not assuming TILE_M = TILE_N
```

### Priority 3: Audit Vector Lane Distribution
```python
# When vector_lane=128 and TILE_N=32 (rectangular)
# How are 128 lanes distributed across 32-wide tile?
# This might cause systematic addressing errors
```

### Priority 4: N-Split Loop Index Generation
```python
# When M-split and N-split both active
# Tile index = base + m_split_idx * TILE_M + n_split_idx * ???
# The ??? calculation might be wrong for rectangular
```

## Verification Steps

1. **Enable debug output** in DMA index calculation
2. **Add assertion** that buffer offsets never exceed allocated size
3. **Trace** a single N-split iteration to see offset calculation
4. **Compare** expected vs actual SRAM addresses for test 1
5. **Run** modified test with debug output enabled

## Commit Recommendation

The current commit 27ede7a should **NOT be reverted** because:
- The stride definition IS correct
- It helps (squares pass)
- But it must be paired with split-iteration fixes

However, the commit message is **misleading** - it claims to "resolve addressing errors" but only resolves stride formula errors. The real addressing errors are in split-iteration DMA code.

## Next Steps

1. Do NOT add more stride fixes
2. DO investigate split-iteration DMA address calculation  
3. DO add comprehensive debug output for DMA offsets
4. DO test with single-split (M-split=1, N-split=1) first
5. DO progressively increase splits to identify failure point
