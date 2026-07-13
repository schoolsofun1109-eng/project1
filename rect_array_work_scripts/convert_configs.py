#!/usr/bin/env python3
"""Convert all top-level configs to the intuitive PE-array / SIMD naming.

Replaces the vpu_* geometry lines with:
    pe_M / pe_N / pe_P   (token x output-ch x input-ch)
    simd_K / simd_bit    (1D SIMD lane count / per-lane bit width)
Values are preserved (pe_N = old vpu_num_lanes, pe_M = systolic_array_height or N,
pe_P = systolic_array_size_k or 1, simd_bit = old vpu_vector_length_bits,
simd_K = N).  IMEM/WMEM/OMEM (SRAM) and everything else are left untouched.
Existing `systolic_array_real_rect` lines are kept (explicit override).
"""
import glob, re, os

def getval(lines, key):
    for ln in lines:
        m = re.match(rf"\s*{re.escape(key)}\s*:\s*([0-9]+)", ln)
        if m: return int(m.group(1))
    return None

for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "cfgdir", "*.yml"))):
    pass  # placeholder; real dir passed via argv

import sys
cfgdir = sys.argv[1]
DROP = ("vpu_num_lanes", "vpu_vector_length_bits", "systolic_array_height", "systolic_array_size_k")
changed = 0
for path in sorted(glob.glob(os.path.join(cfgdir, "*.yml"))):
    with open(path) as f:
        lines = f.readlines()
    # already converted?
    if any(re.match(r"\s*pe_N\s*:", l) for l in lines):
        continue
    N   = getval(lines, "pe_N") or getval(lines, "vpu_num_lanes")
    if N is None:
        continue  # not a core config with lanes
    vlen = getval(lines, "simd_bit") or getval(lines, "vpu_vector_length_bits") or 256
    M   = getval(lines, "systolic_array_height")
    P   = getval(lines, "systolic_array_size_k")
    M = M if M is not None else N
    P = P if P is not None else 1

    block = (
        "# ===== PE Array:  M(token) x N(output-ch) x P(input-ch) =====\n"
        f"pe_M: {M}\n"
        f"pe_N: {N}\n"
        f"pe_P: {P}\n"
        "\n"
        "# ===== 1D SIMD =====\n"
        f"simd_K: {N}\n"
        f"simd_bit: {vlen}\n"
    )

    out, inserted = [], False
    for ln in lines:
        key = ln.split(":")[0].strip() if ":" in ln else ""
        if key in DROP:
            if not inserted:          # drop the first geometry line -> insert block here
                out.append(block); inserted = True
            continue                  # drop remaining old geometry lines
        out.append(ln)
    if not inserted:                  # safety: append if no drop happened
        out.append("\n" + block)
    with open(path, "w") as f:
        f.writelines(out)
    changed += 1
    print(f"  converted {os.path.basename(path)}: pe_M={M} pe_N={N} pe_P={P} simd_K={N} simd_bit={vlen}")

print(f"=== {changed} configs converted ===")
