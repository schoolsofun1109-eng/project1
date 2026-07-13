# External repo patches (GPU 머신 이전 백업)

`gem5`와 `llvm-project`는 shallow clone(depth=1)이라 GitHub로 브랜치 푸시가 불가능했다.
대신 수정 커밋을 패치로 떴다. 베이스 커밋 SHA가 upstream에 그대로 있으므로 아래대로 하면 복원된다.

## gem5

- upstream: https://github.com/PSAL-POSTECH/gem5.git (branch `TorchSim`)
- base commit: `53b8076474698fdb500f1a0077fba387ce040eb4`
- 수정: `src/cpu/minor/{execute.cc,func_unit.hh,BaseMinorCPU.py}`

```bash
git clone -b TorchSim https://github.com/PSAL-POSTECH/gem5.git
cd gem5 && git checkout 53b8076474698fdb500f1a0077fba387ce040eb4
git am < /path/to/patches/gem5.patch
```

## llvm-project

- upstream: https://github.com/PSAL-POSTECH/llvm-project.git
- base commit: `970a927190e8402348cadf9585bf134c8a8c09c2`
- 수정: `mlir/test/lib/Analysis/TestTileOperationGraph.cpp`,
  `mlir/test/lib/Conversion/PyTorchSimToVCIX/TestPyTorchSimToVCIXConversion.cpp`

```bash
git clone https://github.com/PSAL-POSTECH/llvm-project.git
cd llvm-project && git checkout 970a927190e8402348cadf9585bf134c8a8c09c2
git am < /path/to/patches/llvm-project.patch
```

## spike-src

수정 없음 (upstream https://github.com/PSAL-POSTECH/riscv-isa-sim.git 그대로).

## 백업하지 않은 것 (재생성 가능)

- `PyTorchSim/outputs/` (~2GB 시뮬레이션 결과)
- `PyTorchSim/TOGSim/extern/` (서드파티 서브모듈 — `git submodule update --init`)
- 빌드 산출물: `gem5-opt-3d`, `mlir-opt-rect`, `togsim-simulator-*` 바이너리
