#!/usr/bin/env python3
"""Tile dataflow validator for PyTorchSim (SPIKE_DEBUG=1 parser).

Reconstructs, per tile-step and per K-pass:
    DRAM -> SRAM(IMEM/WMEM/OMEM @addr, size) -> SA(push) -> SA(pop, partial) -> DRAM
with sample values (for hand-calc), data sizes, K-pass split, and a routing
check that flags weight/output NOT landing in the dedicated WMEM/OMEM region.

Usage:  SPIKE_DEBUG=1 <run> 2>&1 | python3 tileflow.py [--nval N] [--full]
"""
import sys, re, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--imem", type=lambda x:int(x,0), default=0xd0000000)
ap.add_argument("--wmem", type=lambda x:int(x,0), default=0xd0020000)
ap.add_argument("--omem", type=lambda x:int(x,0), default=0xd0040000)
ap.add_argument("--nval", type=int, default=6)
a = ap.parse_args()

def region(addr):
    if addr >= a.omem: return f"OMEM@+{addr-a.omem:#07x}"
    if addr >= a.wmem: return f"WMEM@+{addr-a.wmem:#07x}"
    return f"IMEM@+{addr-a.imem:#07x}"
def rname(addr):
    return "OMEM" if addr>=a.omem else "WMEM" if addr>=a.wmem else "IMEM"
def sample(vals):
    s=" ".join(f"{v:+.2f}" for v in vals[:a.nval])
    return s+(" ..." if len(vals)>a.nval else "") if vals else "-"
def dsize(m):
    try:
        n=1
        for d in m.get("dims","").split(","): n*=int(d.strip())
        b=n*m.get("esz",4);  return f"{b/1024:.1f}KB" if b>=1024 else f"{b}B"
    except Exception: return "?"

# ---- parse events in order ----
ev=[]; cur=None
for ln in sys.stdin:
    if "=============== MVIN"  in ln: cur={"op":"MVIN"};  ev.append(cur); continue
    if "=============== MVOUT" in ln: cur={"op":"MVOUT"}; ev.append(cur); continue
    m=re.match(r"\[(VPUSH_W|VPUSH_I|VPOP)\] lane\[(\d+)\](.*)",ln)
    if m:
        vals=[]
        for x in m.group(3).split():
            try: vals.append(float(x))
            except ValueError: break
        ev.append({"op":m.group(1),"lane":int(m.group(2)),"vals":vals}); cur=None; continue
    if cur is not None:
        for k,p in (("dram",r"dramAddr: (0x[0-9a-f]+)"),("spad",r"scratchpadAddr: (0x[0-9a-f]+)")):
            mm=re.search(p,ln)
            if mm: cur[k]=int(mm.group(1),16)
        mm=re.search(r"p_dim_size: \(([^)]+)\)",ln);   cur.__setitem__("dims",mm.group(1)) if mm else None
        mm=re.search(r"element_size: (\d+)",ln);        cur.__setitem__("esz",int(mm.group(1))) if mm else None

# ---- split into tile-steps (new step = MVIN after any push/pop) ----
steps=[]; cur=None; pushed=False
for e in ev:
    if e["op"]=="MVIN" and pushed: steps.append(cur); cur=None; pushed=False
    if cur is None: cur=[]
    if e["op"] in ("VPUSH_W","VPUSH_I","VPOP"): pushed=True
    cur.append(e)
if cur: steps.append(cur)

warn=0
for si,st in enumerate(steps,1):
    mvin=[e for e in st if e["op"]=="MVIN"]
    mvout=[e for e in st if e["op"]=="MVOUT"]
    # split the feed/drain sub-sequence into K-passes (a pass ends at VPOP)
    passes=[]; p={"w":0,"i":0,"pop":[],"wv":None,"iv":None}
    for e in st:
        if e["op"]=="VPUSH_W":
            if p["pop"]: passes.append(p); p={"w":0,"i":0,"pop":[],"wv":None,"iv":None}
            p["w"]+=1;  p["wv"]=p["wv"] or (e["vals"] if e["lane"]==0 else None)
        elif e["op"]=="VPUSH_I":
            p["i"]+=1;  p["iv"]=p["iv"] or (e["vals"] if e["lane"]==0 else None)
        elif e["op"]=="VPOP" and e["lane"]==0:
            p["pop"].append(e["vals"])
    if p["w"] or p["pop"]: passes.append(p)

    if not (mvin or passes): continue
    print(f"\n━━━━━ TILE STEP {si}   ({len(passes)} K-pass{'es' if len(passes)!=1 else ''}) ━━━━━")
    for j,m in enumerate(mvin):
        role=["input ","weight","in/w  "][min(j,2)]; exp="IMEM" if j%2==0 else "WMEM"
        fl="" if rname(m.get("spad",0))==exp else f"   ⚠ WMEM 아님(IMEM에 패킹)"
        if fl: warn+=1
        print(f"  LOAD  {role} : DRAM {m.get('dram',0):#x} → {region(m.get('spad',0))}  {dsize(m)}{fl}")
    for pi,p in enumerate(passes,1):
        wv=sample(p["wv"] or []); iv=sample(p["iv"] or [])
        pv=sample(p["pop"][-1] if p["pop"] else [])
        print(f"  │ K-pass {pi}/{len(passes)}: weight×{p['w']} [lane0 {wv}]  input×{p['i']} [lane0 {iv}]")
        print(f"  │            → DRAIN partial [lane0 {pv}]" + ("  (누적)" if pi>1 else ""))
    for u in mvout[:1]:
        exp="OMEM"; fl="" if rname(u.get("spad",0))==exp else f"   ⚠ OMEM 아님(IMEM에 패킹)"
        if fl: warn+=1
        print(f"  STORE output : {region(u.get('spad',0))} → DRAM {u.get('dram',0):#x}  {dsize(u)}"
              + (f"  (mvout×{len(mvout)})" if len(mvout)>1 else "") + fl)

print(f"\n━━━ 라우팅 경고 {warn}건 (weight/output이 전용 WMEM/OMEM 아닌 IMEM에 들어감) ━━━")
