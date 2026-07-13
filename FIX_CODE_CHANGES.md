# MLIR Variable Linking Fix - Exact Code Changes

## File Changed
`00_codes/PyTorchSim/PyTorchSimFrontend/mlir/mlir_gemm_template.py`

## Change 1: W_tile_desc Initialization (Line 163)

### BEFORE (Incorrect)
```python
W_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
```

### AFTER (Correct)
```python
W_tile_desc = mlir_common.MLIRMultiDimTile(W_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
```

### Why This Matters
- **X_tile_size** = [TILE_M, TILE_K] (input matrix dimensions)
- **W_tile_size** = [TILE_K, TILE_N] (weight matrix dimensions)

Using X_tile_size for weight descriptor allocation caused:
- Tile buffer allocated with wrong dimensions
- SRAM offset calculations incorrect
- Data misalignment in rectangular matmul (M != N != K)

### Specific Impact
For rectangular matmul 128×256×256:
- X dimensions: 128×256 → tiles of 64×64
- W dimensions: 256×256 → tiles of 64×64
- **Bug**: W buffer allocated as 64×64 but should use K×N split logic

---

## Change 2: W_stride Assignment (Line 167)

### BEFORE (Incorrect)
```python
W_stride = W.get_layout().stride if N>1 else [Y.get_layout().stride[0], 0]
```

### AFTER (Correct)
```python
W_stride = W.get_layout().stride
```

### Why This Matters
The conditional used Y's stride as fallback for W when N=1:
- For N=1 cases, this gave [Y_stride[0], 0] instead of W's actual stride
- This broken stride calculation cascaded to DRAM address offset calculations
- Nested split iteration contexts (affine.for loops) could not properly compute addresses

### Specific Impact
For rectangular matmul 128×128×1 (edge case):
- N=1 triggers the else branch
- W_stride becomes Y's stride with corrupted dimension [Y_stride[0], 0]
- DMA operations in nested loops access wrong memory locations
- Produces numerical errors (diff > 100)

---

## Change 3: Y_stride Assignment (Line 181)

### BEFORE (Incorrect)
```python
Y_stride = Y.get_layout().stride if N>1 else [Y.get_layout().stride[0], 0]
```

### AFTER (Correct)
```python
Y_stride = Y.get_layout().stride
```

### Why This Matters
Similar to W_stride, but Y is the output matrix.

The conditional applied same fallback logic:
- Should always use correct stride from Y's layout
- Special-case logic for N=1 was unnecessary and incorrect
- Caused output address calculations to be wrong in split contexts

### Specific Impact
For multi-split scenarios (M-split AND N-split):
- Y buffer split across both dimensions
- Each split needs correct stride values
- Fallback logic gave wrong strides, causing buffer overlaps
- Produced numerical mismatches across tile boundaries

---

## Complete Context: Lines 160-185

### BEFORE (With bugs)
```python
155    X_stride = X.get_layout().stride
156    X_idx = [sympy.Symbol("index0") * X_stride[0], sympy.Symbol("index2") * X_stride[1]]
157
158    W_tile_size = [TILE_K, TILE_N]
159    W_tile_stride = [1, TILE_K]
160    W_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)  # BUG 1
161    W_tile_desc.set_tile_size_stride(W_tile_size, W_tile_stride)
162    W_tile_desc.set_name("W_buffer")
163    W_tile_desc.offset = W.get_layout().offset
164    W_stride = W.get_layout().stride if N>1 else [Y.get_layout().stride[0], 0]  # BUG 2
165    W_idx = [sympy.Symbol("index2") * W_stride[0], sympy.Symbol("index1") * W_stride[1]]
166
167    Y_tile_size = [TILE_M, TILE_N] if nr_rdim == 0 else [TILE_N, TILE_M]
168    Y_tile_stride=[1, TILE_M] if nr_rdim == 0 else [TILE_M, 1]
169    Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
170    Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
171    Y_tile_desc.set_name("Y_buffer")
172    Y_stride = Y.get_layout().stride if N>1 else [Y.get_layout().stride[0], 0]  # BUG 3
173    if nr_rdim == 0:
174        Y_idx = [sympy.Symbol("index0") * Y_stride[0], sympy.Symbol("index1") * Y_stride[1]]
175    else:
176        Y_idx = [sympy.Symbol("index1") * Y_stride[1], sympy.Symbol("index0") * Y_stride[0]]
```

### AFTER (Fixed)
```python
155    X_stride = X.get_layout().stride
156    X_idx = [sympy.Symbol("index0") * X_stride[0], sympy.Symbol("index2") * X_stride[1]]
157
158    W_tile_size = [TILE_K, TILE_N]
159    W_tile_stride = [1, TILE_K]
160    W_tile_desc = mlir_common.MLIRMultiDimTile(W_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)  # FIXED
161    W_tile_desc.set_tile_size_stride(W_tile_size, W_tile_stride)
162    W_tile_desc.set_name("W_buffer")
163    W_tile_desc.offset = W.get_layout().offset
164    W_stride = W.get_layout().stride  # FIXED
165    W_idx = [sympy.Symbol("index2") * W_stride[0], sympy.Symbol("index1") * W_stride[1]]
166
167    Y_tile_size = [TILE_M, TILE_N] if nr_rdim == 0 else [TILE_N, TILE_M]
168    Y_tile_stride=[1, TILE_M] if nr_rdim == 0 else [TILE_M, 1]
169    Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
170    Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
171    Y_tile_desc.set_name("Y_buffer")
172    Y_stride = Y.get_layout().stride  # FIXED
173    if nr_rdim == 0:
174        Y_idx = [sympy.Symbol("index0") * Y_stride[0], sympy.Symbol("index1") * Y_stride[1]]
175    else:
176        Y_idx = [sympy.Symbol("index1") * Y_stride[1], sympy.Symbol("index0") * Y_stride[0]]
```

---

## Summary of Changes

| Line | Component | Before | After | Fix Type |
|------|-----------|--------|-------|----------|
| 160 | W_tile_desc init | X_tile_size | W_tile_size | Copy-paste error fix |
| 164 | W_stride | Conditional with Y fallback | Direct assignment | Edge case fix |
| 172 | Y_stride | Conditional with fallback | Direct assignment | Edge case fix |

## Verification Commands

```bash
# View the fix in git
cd /home/sslunder63/project/VQ_NPU_Simulator/00_codes/PyTorchSim
git show 2ce04ce PyTorchSimFrontend/mlir/mlir_gemm_template.py | grep -A 5 -B 5 "W_tile_desc\|W_stride\|Y_stride"

# Verify current code has the fix
grep -n "W_tile_desc = mlir_common.MLIRMultiDimTile(W_tile_size" PyTorchSimFrontend/mlir/mlir_gemm_template.py
grep -n "W_stride = W.get_layout().stride$" PyTorchSimFrontend/mlir/mlir_gemm_template.py
grep -n "Y_stride = Y.get_layout().stride$" PyTorchSimFrontend/mlir/mlir_gemm_template.py
```

Expected output:
```
160:W_tile_desc = mlir_common.MLIRMultiDimTile(W_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
164:W_stride = W.get_layout().stride
172:Y_stride = Y.get_layout().stride
```

All three lines should show the FIXED version (no conditionals, correct variable names).
