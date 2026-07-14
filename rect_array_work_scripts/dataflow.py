#!/usr/bin/env python3
"""Per-operation data-movement view of an NPU run.

Prints one line per DMA in issue order, showing where each tile comes from and
where it lands:

    [   39] MVIN   X  tile(0,0) [128x128] DRAM 0x7f3ab0704000 -> IMEM 0xd0000000

TOGSim (timing) knows the cycle, the tensor name, the DRAM address and the tile
coordinate, but it has no SRAM model at all -- it never sees IMEM/WMEM/OMEM. The
destination comes from the compiler instead: the MLIR kernel binds each argument
to a *_spad global, and the linker places those globals inside the IMEM/WMEM/OMEM
sections. So the two halves are joined here, on the host.

A model compiles to many kernels, each with its own build dir and its own log;
every log names the kernel it belongs to, so they are matched up automatically.

Run the model with TOGSIM_DEBUG_LEVEL=trace, then from the PyTorchSim directory:

    dataflow.py                                  # every kernel of the last run
    dataflow.py --outputs outputs --logs togsim_results
"""

import argparse
import os
import re
import subprocess
import sys
from glob import glob

# Where the sections start, mirroring extension_config.CONFIG_SPAD_INFO: IMEM
# first, then WMEM, then OMEM, packed back to back. Their sizes come from the
# config the run used, so they are options (--imem-size and friends).
IMEM_VADDR = 0xD0000000


def parse_spad_symbols(binary):
    """{symbol: vaddr} for every *_spad global in the linked kernel binary."""
    out = subprocess.run(["nm", "-S", binary], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL).stdout.decode("utf-8", "ignore")
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3].endswith("_spad"):
            syms[parts[3]] = int(parts[0], 16)
    return syms


def parse_arg_to_spad(mlir_path):
    """{arg_index: spad_symbol} from the MLIR kernel.

    The kernel signature names the DRAM tensors and each is bound to a *_spad
    global; the DMAs then move between the two. The Nth kernel argument is the
    Nth `arg` in TOGSim's trace, which is what lets us join them.
    """
    with open(mlir_path) as f:
        text = f.read()

    m = re.search(r"func\.func @kernel\(([^)]*)\)", text)
    if not m:
        return {}
    # "%X: memref<...>, %W: memref<...>" -> ["X", "W", ...]
    arg_names = re.findall(r"%(\w+)\s*:", m.group(1))

    # "%X_buffer = memref.get_global @X_spad" -> X_buffer maps to X_spad
    buf_to_spad = dict(re.findall(r"%(\w+)\s*=\s*memref\.get_global\s+@(\w+_spad)", text))

    # Each dma_start pairs a kernel arg with a buffer, in either direction:
    #   dma_start %X[...], %X_buffer[...]        (mvin)
    #   dma_start %Y_buffer[...], %Y[...]        (mvout)
    arg_to_spad = {}
    for line in text.splitlines():
        if "memref.dma_start" not in line:
            continue
        operands = re.findall(r"%(\w+)\[", line)
        if len(operands) < 2:
            continue
        src, dst = operands[0], operands[1]
        for a, b in ((src, dst), (dst, src)):
            if a in arg_names and b in buf_to_spad:
                arg_to_spad[arg_names.index(a)] = buf_to_spad[b]
    return arg_to_spad


def section_of(vaddr, imem_size, wmem_size, omem_size):
    """Which SRAM a scratchpad vaddr falls in. The linker packs the sections
    back to back starting at IMEM, so the offsets decide it."""
    bounds = [
        ("IMEM", IMEM_VADDR, IMEM_VADDR + imem_size),
        ("WMEM", IMEM_VADDR + imem_size, IMEM_VADDR + imem_size + wmem_size),
        ("OMEM", IMEM_VADDR + imem_size + wmem_size,
                 IMEM_VADDR + imem_size + wmem_size + omem_size),
    ]
    for name, lo, hi in bounds:
        if lo <= vaddr < hi:
            return name, vaddr - lo
    return "SPAD", vaddr - IMEM_VADDR


# One line per DMA issue. MOVIN and MOVOUT print the same fields in a different
# order (see CoreTraceLog.cc::format_dma_inst_issued_detail), so size and stride
# are picked out by name rather than by position.
ISSUED_RE = re.compile(
    r"\[(\d+)\]\[Core (\d+)\]\[INST_ISSUED\s*\]\[INST_ID=(\d+)\]\s+"
    r"(MOVIN|MOVOUT)\s+\(addr_name=(\w+)\s+dram=0x([0-9a-f]+)(.*)$",
    re.MULTILINE,
)
SIZE_RE = re.compile(r"\bsize=\[([\d,]*)\]")
STRIDE_RE = re.compile(r"\bstride=\[([\d,-]*)\]")
# TOGSim prints each tensor's DRAM base once, at parse time.
BASE_RE = re.compile(r"Address Attribute key: (\w+) address: 0x([0-9a-f]+)")
# Each log covers one kernel and names it, which is what ties the log to the
# build dir whose MLIR/binary hold the SRAM placement. A model like Llama
# compiles to many kernels, each with its own log, so this must not be guessed.
KERNEL_RE = re.compile(r"Enqueued kernel_id: \d+, tog_path: \S*?outputs/(\w+)/[^,]*,\s*operation: (\S+)")


def tile_coord(dram, base, tile_size, tile_stride):
    """Which tile of the tensor this DMA touches.

    The offset from the tensor's base, divided by the tile's extent along each
    axis, is the tile index. The offset is in *elements*: TOGSim builds the
    instruction address as base + inner_product(loop_idx, loop_stride), and the
    strides it gets from the TOG graph are element counts, matching the affine
    maps in the MLIR (e.g. `256*index0 + index2` for a 384x256 X).

    TOGSim's own tag_idx cannot be used for this -- it is a cache-tag artifact
    and stays [0,0] for every tile.
    """
    if not base or not tile_size or not tile_stride:
        return ""
    elems = dram - base
    coords = []
    for size, stride in zip(tile_size, tile_stride):
        if stride == 0 or size == 0:
            coords.append(0)
            continue
        coords.append((elems // stride) // size)
        elems %= stride
    return ",".join(str(c) for c in coords)


def report_kernel(log, outputs_dir, sizes):
    """Print the data movement for the one kernel this log covers."""
    with open(log) as f:
        text = f.read()

    m = KERNEL_RE.search(text)
    if not m:
        return False
    op_name, run_dir = m.group(2), os.path.join(outputs_dir, m.group(1))

    binary = os.path.join(run_dir, "validation_binary")
    mlirs = [p for p in glob(os.path.join(run_dir, "*.mlir"))
             if not p.endswith("_llvm.mlir")]
    if not os.path.exists(binary) or not mlirs:
        print(f"# {os.path.basename(run_dir)}: no binary/mlir, skipped")
        return False

    spad_syms = parse_spad_symbols(binary)
    arg_to_spad = parse_arg_to_spad(mlirs[0])
    bases = {n: int(a, 16) for n, a in BASE_RE.findall(text)}

    print(f"=== kernel {os.path.basename(run_dir)}  ({op_name}) ===")
    print("    tensors: " + ", ".join(
        f"arg{i}->{s}@0x{spad_syms.get(s, 0):x}" for i, s in sorted(arg_to_spad.items())))

    rows = []
    for cyc, core, iid, op, name, dram, rest in ISSUED_RE.findall(text):
        idx = int(name[3:]) if name.startswith("arg") and name[3:].isdigit() else None
        spad = arg_to_spad.get(idx)
        vaddr = spad_syms.get(spad, 0) if spad else 0
        sect, _ = section_of(vaddr, *sizes)
        tensor = spad[:-5] if spad else name  # X_spad -> X

        m_size, m_stride = SIZE_RE.search(rest), STRIDE_RE.search(rest)
        tile_size = [int(v) for v in m_size.group(1).split(",") if v] if m_size else []
        tile_stride = [int(v) for v in m_stride.group(1).split(",") if v] if m_stride else []
        coord = tile_coord(int(dram, 16), bases.get(name, 0), tile_size, tile_stride)

        rows.append((int(cyc), op, tensor, coord,
                     "x".join(str(s) for s in tile_size), dram, sect, vaddr))

    rows.sort()
    for cyc, op, tensor, coord, shape, dram, sect, vaddr in rows:
        mv = "MVIN " if op == "MOVIN" else "MVOUT"
        src, dst = f"DRAM 0x{dram}", f"{sect} 0x{vaddr:x}"
        if op == "MOVOUT":
            src, dst = dst, src
        tile = f"tile({coord})" if coord else ""
        print(f"[{cyc:6}] {mv} {tensor:<6} {tile:<12} [{shape:>9}]  {src:>22} -> {dst}")

    if not rows:
        print("    (no DMA -- was the run made with TOGSIM_DEBUG_LEVEL=trace?)")
    print()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs",
                    help="directory holding the per-kernel build dirs (default: outputs)")
    ap.add_argument("--logs", default="togsim_results",
                    help="directory holding the TOGSim logs (default: togsim_results)")
    ap.add_argument("--imem-size", type=lambda x: int(x, 0), default=128 * 1024)
    ap.add_argument("--wmem-size", type=lambda x: int(x, 0), default=128 * 1024)
    ap.add_argument("--omem-size", type=lambda x: int(x, 0), default=128 * 1024)
    args = ap.parse_args()

    logs = sorted(glob(os.path.join(args.logs, "*.log")))
    if not logs:
        sys.exit(f"no TOGSim logs in {args.logs}/ -- run with TOGSIM_DEBUG_LEVEL=trace")

    sizes = (args.imem_size, args.wmem_size, args.omem_size)
    if not sum(report_kernel(log, args.outputs, sizes) for log in logs):
        sys.exit("no kernel found in any log -- run with TOGSIM_DEBUG_LEVEL=trace")


if __name__ == "__main__":
    main()
