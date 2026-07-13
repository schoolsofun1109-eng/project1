#!/usr/bin/env python3
"""Combined dataflow + full-stat tracer for PyTorchSim runs.

Wraps a test command: runs it with SPIKE_DEBUG=1, then prints, per tile-step,
BOTH the SRAM<->SA dataflow (same logic as tileflow.py: DRAM->SRAM(region@addr)
->SA(push)->SA(pop)->DRAM with sample values) AND -- unlike tileflow.py's
summarized [METRICS] log lines -- the FULL raw TOGSim per-core stat block
(instruction counts, per-array active/idle cycles, DMA active/idle+BW,
vector unit active/idle, per-channel DRAM BW/util) for every kernel run
during the command, verbatim, not regex-summarized. This exists because the
summarized METRICS line has already hidden real bugs once (multi-core array
index collision) -- this tool intentionally shows everything so a human (or
Claude) can eyeball raw numbers instead of trusting a summary.

Usage:
  python3 simtrace.py -- <command...>
  e.g. python3 simtrace.py -- python3 tests/test_matmul.py

Run inside the container where TOGSIM_CONFIG / paths are already set up.
"""
import sys, re, subprocess, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--imem", type=lambda x: int(x, 0), default=0xd0000000)
ap.add_argument("--wmem", type=lambda x: int(x, 0), default=0xd0020000)
ap.add_argument("--omem", type=lambda x: int(x, 0), default=0xd0040000)
ap.add_argument("--nval", type=int, default=6)
ap.add_argument("--num-arrays", type=int, default=0,
                 help="systolic arrays per core (for SA[idx] round-robin annotation); "
                      "0 = auto-read num_systolic_array_per_core from $TOGSIM_CONFIG")
ap.add_argument("--no-dataflow", action="store_true", help="skip the tile dataflow section, only show stats")
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
if a.cmd and a.cmd[0] == "--":
    a.cmd = a.cmd[1:]
if not a.cmd:
    print("usage: simtrace.py -- <command...>", file=sys.stderr)
    sys.exit(1)


def _detect_num_arrays():
    if a.num_arrays > 0:
        return a.num_arrays
    cfg_path = os.environ.get("TOGSIM_CONFIG")
    if cfg_path and os.path.isfile(cfg_path):
        try:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            return int(cfg.get("num_systolic_array_per_core", 1))
        except Exception:
            pass
    return 1


NUM_ARRAYS = _detect_num_arrays()


def region(addr):
    if addr >= a.omem: return f"OMEM@+{addr-a.omem:#07x}"
    if addr >= a.wmem: return f"WMEM@+{addr-a.wmem:#07x}"
    return f"IMEM@+{addr-a.imem:#07x}"


def rname(addr):
    return "OMEM" if addr >= a.omem else "WMEM" if addr >= a.wmem else "IMEM"


def nbytes(m):
    """Raw byte count for a MVIN/MVOUT event, or None if dims/esz missing."""
    try:
        n = 1
        for d in m.get("dims", "").split(","):
            n *= int(d.strip())
        return n * m.get("esz", 4)
    except Exception:
        return None


def range_str(m):
    """'<REGION>@+start ~ +end (size)' -- the actual address span written, not
    just a start offset, so overlap/overflow (e.g. a tile bigger than its
    region's budget) is visible directly instead of requiring separate math."""
    addr = m.get("spad", 0)
    b = nbytes(m)
    if b is None:
        return region(addr)
    end = addr + b
    base = a.omem if addr >= a.omem else a.wmem if addr >= a.wmem else a.imem
    sz = f"{b/1024:.1f}KB" if b >= 1024 else f"{b}B"
    return f"{rname(addr)}@+{addr-base:#07x}~+{end-base:#07x} ({sz})"


def sample(vals):
    s = " ".join(f"{v:+.2f}" for v in vals[:a.nval])
    return s + (" ..." if len(vals) > a.nval else "") if vals else "-"


def print_dataflow(lines):
    ev = []
    cur = None
    for ln in lines:
        if "=============== MVIN" in ln:
            cur = {"op": "MVIN"}; ev.append(cur); continue
        if "=============== MVOUT" in ln:
            cur = {"op": "MVOUT"}; ev.append(cur); continue
        m = re.match(r"\[(VPUSH_W|VPUSH_I|VPOP)\] lane\[(\d+)\](.*)", ln)
        if m:
            vals = []
            for x in m.group(3).split():
                try:
                    vals.append(float(x))
                except ValueError:
                    break
            ev.append({"op": m.group(1), "lane": int(m.group(2)), "vals": vals})
            cur = None
            continue
        if cur is not None:
            for k, p in (("dram", r"dramAddr: (0x[0-9a-f]+)"), ("spad", r"scratchpadAddr: (0x[0-9a-f]+)")):
                mm = re.search(p, ln)
                if mm:
                    cur[k] = int(mm.group(1), 16)
            mm = re.search(r"p_dim_size: \(([^)]+)\)", ln)
            if mm:
                cur["dims"] = mm.group(1)
            mm = re.search(r"element_size: (\d+)", ln)
            if mm:
                cur["esz"] = int(mm.group(1))

    steps = []
    cur = None
    pushed = False
    for e in ev:
        if e["op"] == "MVIN" and pushed:
            steps.append(cur); cur = None; pushed = False
        if cur is None:
            cur = []
        if e["op"] in ("VPUSH_W", "VPUSH_I", "VPOP"):
            pushed = True
        cur.append(e)
    if cur:
        steps.append(cur)

    warn = 0
    # NOTE on "which systolic array": each K-pass here is exactly one PRELOAD
    # node (all weight pushes) + one MATMUL node (input pushes + drain), per
    # the FSM in TestTileOperationGraph.cpp:667-728 -- confirmed against
    # golden cases (N K-passes -> COMP inst_count GEMM=2N). TOGSim assigns
    # each node to a systolic array via round-robin (Core.cc:52-53), BUT the
    # round-robin order follows RUNTIME dispatch order, not static K-pass
    # order -- a later pass's PRELOAD can be prefetched/dispatched before an
    # earlier pass's MATMUL finishes (pipelining), so which array a given
    # K-pass lands on isn't derivable from this Spike-side trace alone (that
    # would need TOGSim's own scheduler state). The real, ground-truth
    # per-array split IS available below in "TOGSim raw stats" -> "Systolic
    # array [i] ... active_cycles" -- trust that, not a per-K-pass guess here.
    for si, st in enumerate(steps, 1):
        mvin = [e for e in st if e["op"] == "MVIN"]
        mvout = [e for e in st if e["op"] == "MVOUT"]
        passes = []
        p = {"w": 0, "i": 0, "pop": [], "wv": None, "iv": None}
        for e in st:
            if e["op"] == "VPUSH_W":
                if p["pop"]:
                    passes.append(p); p = {"w": 0, "i": 0, "pop": [], "wv": None, "iv": None}
                p["w"] += 1
                p["wv"] = p["wv"] or (e["vals"] if e["lane"] == 0 else None)
            elif e["op"] == "VPUSH_I":
                p["i"] += 1
                p["iv"] = p["iv"] or (e["vals"] if e["lane"] == 0 else None)
            elif e["op"] == "VPOP" and e["lane"] == 0:
                p["pop"].append(e["vals"])
        if p["w"] or p["pop"]:
            passes.append(p)

        if not (mvin or passes):
            continue
        print(f"\n━━━━━ TILE STEP {si}   ({len(passes)} K-pass{'es' if len(passes) != 1 else ''}) ━━━━━")
        for j, m in enumerate(mvin):
            role = ["input ", "weight", "in/w  "][min(j, 2)]
            exp = "IMEM" if j % 2 == 0 else "WMEM"
            fl = "" if rname(m.get("spad", 0)) == exp else "   ⚠ WMEM 아님(IMEM에 패킹)"
            if fl:
                warn += 1
            print(f"  LOAD  {role} : DRAM {m.get('dram', 0):#x} → {range_str(m)}{fl}")
        for pi, p in enumerate(passes, 1):
            wv = sample(p["wv"] or [])
            iv = sample(p["iv"] or [])
            pv = sample(p["pop"][-1] if p["pop"] else [])
            print(f"  │ K-pass {pi}/{len(passes)} [1 PRELOAD + 1 MATMUL node -> some SA[i], see raw stats]: "
                  f"weight×{p['w']} [lane0 {wv}]  input×{p['i']} [lane0 {iv}]")
            print(f"  │            → DRAIN partial [lane0 {pv}]" + ("  (누적)" if pi > 1 else ""))
        for u in mvout[:1]:
            exp = "OMEM"
            fl = "" if rname(u.get("spad", 0)) == exp else "   ⚠ OMEM 아님(IMEM에 패킹)"
            if fl:
                warn += 1
            print(f"  STORE output : {range_str(u)} → DRAM {u.get('dram', 0):#x}"
                  + (f"  (mvout×{len(mvout)})" if len(mvout) > 1 else "") + fl)
    if warn:
        print(f"\n━━━ 라우팅 경고 {warn}건 ━━━")
    return warn


def print_full_stats(log_path):
    if not os.path.isfile(log_path):
        print(f"  (log not found: {log_path})")
        return
    with open(log_path) as f:
        content = f.read()
    # Print from the per-channel DRAM BW/util summary through the final
    # "Total execution cycles" line, verbatim -- full per-core detail, not
    # the lossy regex summary in simulator.py's [METRICS] output. Skips the
    # "=== DRAM statistics ===" preamble on purpose: that's Ramulator2's own
    # internal config+row-buffer-hit/miss dump, repeated per channel (huge,
    # 16-32x) and not useful for checking NPU-level correctness -- everything
    # in it that matters (achieved GB/s, % utilization) is restated per
    # channel in the "[DRAM] channel N | ..." lines this starts from.
    m = re.search(
        r"(\[DRAM\] Per-channel average bandwidth.*?Total execution cycles: \d+)",
        content, re.S,
    )
    if m:
        for line in m.group(1).splitlines():
            line = re.sub(r"^\[.*?\]\s*\[info\]\s*", "", line)
            print("  " + line)
    else:
        print("  (no stat block found in log)")


# Matches TOGSim's per-instruction trace lines (spdlog::trace, enabled via
# TOGSIM_DEBUG_LEVEL=trace): "[timestamp] [trace] [cycle][Core N][TAG][INST_ID=X] message"
# A few event kinds (e.g. "TOG async DMA response") omit the [TAG] bracket.
_CYCLE_LINE_RE = re.compile(
    r"^\[.*?\]\s*\[trace\]\s*\[(\d+)\]\[Core (\d+)\](?:\[([^\]]*)\])?(?:\[INST_ID=(\d+)\])?\s*(.*)$"
)
_CYCLE_TRACE_CAP = 800  # keep output bounded for large kernels (many DMA/tile events)


_COMP_RE = re.compile(r"COMP \(compute_type=(\d+) compute_cycle=(\d+) overlapping_cycle=(\d+)\)")


def print_cycle_trace(log_path):
    """Per-cycle 'what happened' trace: every instruction's issue/finish cycle,
    with compute_cycle/overlapping_cycle for COMP ops (this is the raw source
    of the active_cycles accounting -- see Core.cc:265-277). Needs
    TOGSIM_DEBUG_LEVEL=trace, which main() sets automatically.

    Each line is also annotated with the RUNNING SA/VU active/idle cycle total
    as of that exact cycle -- i.e. following the cycle numbers down this trace
    IS watching Core.cc's active_cycles/idle_cycles counters accumulate live,
    the same numbers that show up as a single final line in "TOGSim raw
    stats" above. `id=N` is TOGSim's global_inst_id: a unique per-instruction
    number assigned in dispatch order, used here (and by TOGSim itself) to
    match an INST_ISSUED line to its later INST_FINISHED line for the *same*
    instruction -- most events carry one, a few pure scheduling events
    (TILE_SCHEDULED) don't since they're not about one specific instruction.

    The running total is reconstructed in Python, not printed live by TOGSim
    itself (it only prints one final summary line) -- from two facts verified
    directly against Core.cc and cross-checked against real runs:
      1) a COMP instruction's net active contribution, regardless of how much
         of its overlapping_cycle actually got hidden behind a neighbor (that
         depends on runtime queueing -- Core.cc:266-277), always collapses to:
           SA (compute_type 1/2): compute_cycle - overlapping_cycle
           VU (compute_type 0):   compute_cycle - overlapping_cycle + 1
         (the "+1": vu_cycle()'s active++ sits outside the finish-check --
         Core.cc:66 -- so it counts its own finish tick as active; sa_cycle()'s
         sits inside the else branch -- Core.cc:96 -- so it doesn't.)
      2) active_so_far + idle_so_far == core_cycles_elapsed_so_far always
         holds (every core cycle is classified as exactly one or the other),
         so idle_so_far is just "this cycle number - active_so_far".
    Only meaningful for num_systolic_array_per_core == 1 -- with >1 arrays,
    TOGSim's round-robin array assignment isn't recoverable from this log, so
    SA numbers would be the combined pool across all arrays, not per-array.
    """
    if not os.path.isfile(log_path):
        return
    with open(log_path) as f:
        content = f.read()
    rows = []
    for line in content.splitlines():
        m = _CYCLE_LINE_RE.match(line)
        if m:
            cycle, core, tag, inst_id, rest = m.groups()
            rows.append((int(cycle), core, (tag or "").strip(), inst_id, rest.strip()))
    if not rows:
        print("  (no cycle trace found -- set TOGSIM_DEBUG_LEVEL=trace, or it wasn't picked up)")
        return
    rows.sort(key=lambda r: r[0])
    shown = rows[:_CYCLE_TRACE_CAP]

    sa_active = vu_active = 0
    for cycle, core, tag, inst_id, rest in shown:
        cm = _COMP_RE.search(rest)
        if cm and "FINISHED" in tag:
            ctype, ccyc, ocyc = int(cm.group(1)), int(cm.group(2)), int(cm.group(3))
            if ctype in (1, 2):
                sa_active += ccyc - ocyc
            else:
                vu_active += ccyc - ocyc + 1
        sa_idle, vu_idle = cycle - sa_active, cycle - vu_active
        idpart = f" id={inst_id}" if inst_id else ""
        tagpart = f" {tag}" if tag else ""
        print(f"  cycle {cycle:>6}  Core{core}{tagpart}{idpart}  {rest}"
              f"   [SA누적 active={sa_active} idle={sa_idle} util={sa_active*100/cycle:.2f}%]"
              f"  [VU누적 active={vu_active} idle={vu_idle} util={vu_active*100/cycle:.2f}%]")
    if len(rows) > _CYCLE_TRACE_CAP:
        print(f"  ... ({len(rows) - _CYCLE_TRACE_CAP} more events truncated; "
              f"read {log_path} directly for the full trace)")


# Framework noise to drop when showing "what the command actually printed":
# timestamped logger lines, eager-mode fallback notices, Spike/TOGSim/gem5
# progress tickers, and Spike's per-instruction SPIKE_DEBUG=1 trace (MVIN/
# MVOUT/VPUSH/VPOP blocks, register/config dumps, per-element Val(hex)/
# Val(float) lines) -- all of that is already summarized by the dataflow
# trace below, so repeating it here would just be noise. What's left is the
# test script's own print()s: which case ran, shapes, and the result (e.g.
# "GOLDEN MATCH = True").
_NOISE_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} "                       # timestamped logger lines
    r"|\[Eager Mode\]"
    r"|Running simulation:"
    r"|^\[DEBUG SPIKE\]|^\[STARTUP\]"
    r"|^===============\s*(MVIN|MVOUT)"            # Spike event section headers
    r"|^========\s*CONFIG"                         # Spike CONFIG/CONFIG2/CONFIG3 headers
    r"|^={5,}\s*COMPUTE|^-{5,}\s*\w+\s*-{5,}"      # Spike matrix-dump section headers
    r"|^\[MOVIN\]|^\[MOVOUT\]|^\[VPUSH|^\[VPOP\]"   # per-element instruction trace
    r"|^- "                                        # indented field: value debug lines
    r"|^RS1:|^RS2:"                                # raw register dumps
    r"|^(Number of vectorlane|Scratchpad base|Kernel addr|MEM >>|Base path):?"  # Spike startup banner
    r"|^(Instruction configs|DMA buffer configs|Block configs|Load data from mm|Store data to mm):"
    r"|^(dim_size|element_size|vlane_split_axis|vlane_stride|indirect_mode|mm_stride|spad_stride|p_dim_size|p_mm_stride|p_spad_stride)\s*="
)


def _is_raw_number_dump(line):
    """Catch-all for Spike's whole-matrix debug dumps (rows of bare floats)
    that no fixed header pattern covers -- if most tokens on the line parse
    as a float, it's data, not a message."""
    tokens = line.split()
    if len(tokens) < 4:
        return False
    numeric = 0
    for t in tokens:
        try:
            float(t)
            numeric += 1
        except ValueError:
            pass
    return numeric / len(tokens) > 0.7


def main():
    env = os.environ.copy()
    env["SPIKE_DEBUG"] = "1"
    env["TOGSIM_DEBUG_LEVEL"] = "trace"
    proc = subprocess.run(a.cmd, capture_output=True, text=True, env=env)
    out = proc.stdout + proc.stderr
    lines = out.splitlines()

    print(f"[simtrace] NUM_ARRAYS={NUM_ARRAYS} (num_systolic_array_per_core"
          f"{' from $TOGSIM_CONFIG' if a.num_arrays == 0 else ' via --num-arrays'})")
    print(f"╔{'═'*70}")
    print(f"║ Test output (which case ran + its result; framework noise filtered)")
    print(f"╚{'═'*70}")
    for line in lines:
        if line.strip() and not _NOISE_RE.search(line) and not _is_raw_number_dump(line):
            print(line)

    if not a.no_dataflow:
        print_dataflow(lines)

    log_paths = re.findall(r'Simulation log is stored to "([^"]+)"', out)
    print(f"\n╔{'═'*70}")
    print(f"║ TOGSim raw stats: {len(log_paths)} kernel run(s)")
    print(f"╚{'═'*70}")
    for i, lp in enumerate(log_paths, 1):
        print(f"\n--- kernel run {i}/{len(log_paths)}: {os.path.basename(lp)} ---")
        print_full_stats(lp)
        print(f"\n  ── cycle-by-cycle trace (매 줄에 SA/VU 누적 active/idle/util 포함) ──")
        print_cycle_trace(lp)

    if proc.returncode != 0:
        print(f"\n[simtrace] command exited {proc.returncode}. Last 30 lines:")
        print("\n".join(lines[-30:]))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
