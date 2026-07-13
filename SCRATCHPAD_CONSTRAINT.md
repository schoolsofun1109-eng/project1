# Scratchpad Memory Constraint

## Current Limitation (Spike Emulator)

When using IMEM/WMEM/OMEM separated memory:
```
sum(imem_size, wmem_size, omem_size) ≤ scratchpad_size / 2
```

### Example (128KB scratchpad)
```yaml
scratchpad_size: 128KB
constraint: imem + wmem + omem ≤ 64KB

✅ Valid:
  imem: 32KB, wmem: 16KB, omem: 16KB  (sum=64KB)
  imem: 64KB, wmem: 0KB, omem: 0KB    (sum=64KB)

❌ Invalid:
  imem: 64KB, wmem: 32KB, omem: 32KB  (sum=128KB > 64KB)
  imem: 40KB, wmem: 30KB, omem: 10KB  (sum=80KB > 64KB)
```

## Reason

Spike internally treats scratchpad offset as signed value with bitwidth=log2(scratchpad_size).
When offset ≥ scratchpad_size/2, sign bit is set → wrong paddr → silent data corruption.

## Workaround

Keep total memory ≤ 64KB (for 128KB scratchpad):
```yaml
imem_num_banks: 2
imem_sram_bitwidth: 256
imem_sram_depth: 256  # 32KB

wmem_num_banks: 1
wmem_sram_bitwidth: 256
wmem_sram_depth: 256  # 16KB

omem_num_banks: 1
omem_sram_bitwidth: 256
omem_sram_depth: 256  # 16KB

# Total = 32 + 16 + 16 = 64KB ✓
```

## Long-term Fix

Modify Spike emulator to separate scratchpad_size (lane stride) from vaddr range check,
or use fixed-width unsigned offset calculation.
