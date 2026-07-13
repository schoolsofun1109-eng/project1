# IMEM / WMEM / OMEM 분리 메모리 모델 전환 — 코드 변경 분석 보고서

> 작성일: 2026-07-01
> 목적: PyTorchSim의 **lane 기반 통합 SPAD** 모델을 **IMEM/WMEM/OMEM 분리 SRAM bank** 모델로 전환하기 위해
> 코드 전체에서 바꿔야 할 모든 지점을 정리.

---

## ★ 구현 현황 (2026-07-01 반영)

**완료 (backward-compatible: 기존 config엔 영향 없음, 새 config `configs/imem_wmem_omem_c1.yml`에서만 활성):**
- [A] `extension_config.py`: `CONFIG_IMEM/WMEM/OMEM` 구현(용량 = banks×bitwidth×depth/8, base 주소 포함). yml에 키 없으면 `None`→기존 `.spad` fallback.
- [A] `mlir_common.py BaseMLIRHardwareInfo`: `imem/wmem/omem_info` + `separate_memory_enabled()` + `get_mem_target(role)` 헬퍼.
- [B] `mlir_codegen_backend.py`: `allocate_sram_buffer`/`get_scratchpad_buffer`가 `mem_role`로 `.imem/.wmem/.omem` 섹션 라우팅.
- [B] `mlir_template.py def_sram_buffer`: `mem_role` 파라미터 추가.
- [B] 템플릿: gemm/bmm/conv(4종)/sdpa → X=input, W=weight, Y=output, Bias=WMEM, SDPA key/value=WMEM, 임시=OMEM.
- [C] `extension_codecache.py`: 링커 `--section-start=.imem/.wmem/.omem` + 메모리별 overflow 체크.
- [C] `mlir_caller_codegen.py get_spad_size`: 섹션명 인자화.
- [C] `mlir_codegen_backend.py`: `.imem/.wmem/.omem` end 앵커 심볼(빈 섹션도 생성).
- 새 config: `configs/imem_wmem_omem_c1.yml` (사수님 예시값 8banks×256b×512).

**결정/보류:**
- [D] Spike: **변경 불필요**. `.imem/.wmem/.omem`은 특수 scratchpad(.spad)가 아닌 일반 로드 섹션이며 주소가 spike main-mem 매핑 범위(0x80000000+100GB) 안 → pk가 정상 매핑. 기능검증은 값만 맞으면 OK. **(내일 실제 spike 실행으로 확인 필요)**
- [E] gem5 per-role timing: **gem5 fork 필요, 이번엔 미구현**. 이유 2가지: ①gem5용 헤더(`gem5_global_var.h`)엔 애초에 섹션 속성이 없어 gem5는 전부 단일 메모리로 봄. ②gem5 SE 모드는 vaddr→paddr 프레임을 자체 할당해서 스크립트에서 물리주소 범위로 나눠도 버퍼가 안 떨어짐. → `.spad`처럼 fork의 scratchpad 메커니즘을 IMEM/WMEM/OMEM으로 확장해야 정확. `script_systolic.py`에 설명 주석만 추가(동작 보존).
- [G] 3D PE(M×N×P): 별도 과제(gem5/llvm fork).

**검증 완료:** 전 파일 `py_compile` 통과, 새/기존 config 로드 확인, 역할 라우팅(input→.imem 등) 확인, 기존 config는 전부 `.spad` fallback 확인.

**내일 할 것:** Docker 컨테이너에서 `configs/imem_wmem_omem_c1.yml`로 작은 GEMM end-to-end 실행 → spike 기능검증 통과 확인 → 섹션 크기/overflow 동작 확인.

---

## 0. 한 줄 요약

현재 PyTorchSim은 **입력·가중치·출력을 구분하지 않고** 하나의 `.spad` 섹션(단일 scratchpad)에 넣고,
`vector_lane` 개수로 자동 분할한다.
목표 NPU core는 **IMEM(입력) / WMEM(가중치) / OMEM(출력)** 3개의 독립 SRAM으로 나뉘고,
각 메모리가 `bank 수 × (bitwidth × depth)` 로 정의된다.
→ "데이터를 어디에 저장하느냐"를 lane이 아니라 **역할별 bank 메모리**로 바꾸는 작업.

---

## 1. 현재 구조 (AS-IS)

### 1.1 메모리 모델
- **단일 SPAD**: 모든 SRAM 버퍼(X/W/Y/Bias/임시버퍼)가 `__attribute__((section(".spad")))` 하나로 들어감.
- **주소**: linker가 `.spad`를 `spad_vaddr = 0xD0000000` 한 군데에 배치.
- **크기**: `spad_size = vpu_spad_size_kb_per_lane × 1024` (per-lane) × `vector_lane`.
- **분할 방식**: `VectorLaneMapping`이 tile을 `vlane_split_axis` 축으로 `vector_lane`개 lane에 쪼갬.

### 1.2 각 시뮬레이터의 역할 (매우 중요 — 어디를 고쳐야 하는지 결정)

| 단계 | 도구 | 위치 | 메모리를 어떻게 다루나 | 소스 위치 |
|------|------|------|----------------------|-----------|
| 코드 생성 | MLIR 패스 | `mlir-opt` | 컴파일 시 tile/DMA 생성. `.spad`만 앎 | **외부 fork** (llvm-project) |
| 기능 검증 | spike | `run_spike` | `--scratchpad-*` **단일 영역** 1개 | **외부 fork** (riscv-isa-sim) |
| 사이클 계산 | gem5 | `script_systolic.py` | `MultiBankMemorySystem`(현재 bank=1), `SystolicArray` FU | **binary는 외부 fork, 설정 스크립트는 repo 내부** |
| 타일 스케줄 | TOGSim | `TOGSim/src/*.cc` | **DRAM 주소만** 모델링, SRAM은 암묵적 | **repo 내부 C++** |

**핵심 결론**:
- SRAM의 **실제 timing(대역폭·bank 충돌)** 은 → **gem5 (`script_systolic.py`)** 에서 결정된다. ← SRAM 작업의 주 전장
- **기능 정확성**은 → spike. 단일 scratchpad로도 3영역이 겹치지 않게 배치되면 동작 가능.
- **주소 배치**는 → linker 섹션(`.spad`) + Python `spad_info`.
- **역할 구분(입력/가중치/출력)** 정보는 → **Python 템플릿에 이미 다 있음** (`def_dma_op("MVIN","X"/"W"...)`, `prologue_info["input_dram_var"]/["weight_dram_var"]`).

---

## 2. 목표 구조 (TO-BE)

```
IMEM (입력 활성화)   : num_banks × bitwidth × depth,  base_vaddr = A
WMEM (가중치)        : num_banks × bitwidth × depth,  base_vaddr = B
OMEM (출력)          : num_banks × bitwidth × depth,  base_vaddr = C

PE Array : M(token) × N(och) × P(ich)   ← 3D
1D SIMD  : K개 PE                        ← 현재 vector_lane 에 해당
```

역할 라우팅:
- 입력 활성화(X, query, key, value) → **IMEM**
- 가중치(W) → **WMEM**
- 출력(Y, out) → **OMEM**
- Bias / 임시버퍼 → **설계 결정 필요** (아래 6장)

---

## 3. 바꿔야 할 모든 지점 (파일·라인 단위)

### [A] 설정 계층 (Config) — 난이도 하

#### A-1. `PyTorchSimFrontend/extension_config.py`  (라인 77~99)
- **이미 주석으로 초안이 있음**. `CONFIG_IMEM/WMEM/OMEM` (num_banks, bitwidth, depth, capacity_kb) 계산 로직.
- 할 일: 주석 해제 + 실제 반환하도록 `__getattr__`에 편입.
- `capacity_kb = num_banks × bitwidth × depth / 8 / 1024` 공식 확인.

#### A-2. `configs/*.yml` (예: `systolic_ws_128x128_c1_simple_noc_tpuv3.yml`)
- 현재 존재: `vpu_num_lanes`, `pe_array_m/n/k`, `vpu_spad_size_kb_per_lane`.
- 추가 필요:
  ```yaml
  imem_num_banks:  ...
  imem_sram_bitwidth: ...
  imem_sram_depth: ...
  wmem_num_banks:  ...
  wmem_sram_bitwidth: ...
  wmem_sram_depth: ...
  omem_num_banks:  ...
  omem_sram_bitwidth: ...
  omem_sram_depth: ...
  ```
- 여러 config 파일에 동일하게 넣어야 함(사용하는 yml만이라도).

#### A-3. `PyTorchSimFrontend/mlir/mlir_common.py`  `BaseMLIRHardwareInfo` (라인 613~625)
- 현재: `self.spad_info = extension_config.CONFIG_SPAD_INFO` (단일).
- 추가: `self.imem_info / wmem_info / omem_info = extension_config.CONFIG_IMEM/WMEM/OMEM`.
- 각 info에 base_vaddr / base_paddr / size 포함시켜야 함(현재 spad_info 형식 참고).

---

### [B] Python 코드 생성 — SRAM 버퍼 라우팅 (난이도 중, **가장 중요한 in-repo 작업**)

#### B-1. `PyTorchSimFrontend/mlir/mlir_codegen_backend.py` `allocate_sram_buffer` (라인 1434~1458)
- **현재 모든 버퍼가 `.spad`로 감** (라인 1452):
  ```python
  self.header.writeline(f"{c_type} {new_name}[{tile_size // self.vector_lane}] __attribute__ ((section(\".spad\")));")
  ```
- 할 일: 인자로 **메모리 역할(mem_role: imem/wmem/omem)** 을 받아서 섹션명을 분기:
  ```python
  section = {".imem",".wmem",".omem"}[mem_role]
  ... section(\"{section}\") ...
  ```
- 버퍼 이름도 `buf{N}_imem` 식으로 접두어를 나눠 충돌 방지 고려.

#### B-2. `mlir_codegen_backend.py` `get_scratchpad_buffer` (라인 1460~1471)
- `allocate_sram_buffer` 호출부. mem_role 전달 경로 추가.

#### B-3. `mlir_codegen_backend.py` `index_expr`의 인덱스용 spad 버퍼 (라인 869)
- `index_expr_*_spad` → 임시 인덱스 버퍼. 어느 메모리에 둘지 결정(보통 OMEM 또는 별도).

#### B-4. `PyTorchSimFrontend/mlir/mlir_template.py` `def_sram_buffer` (라인 1021~1030)
- **역할 라우팅의 최적 진입점**. `dram_name`("X"/"W"/"Y"/"Bias"/"query"...)을 이미 받음.
- `dram_name` → mem_role 매핑 테이블을 만들어 `allocate_sram_buffer`에 전달.
- 단, `dram_name`만으로는 부족(예: SDPA는 query/key/value 전부 입력).
  → 템플릿에서 명시적으로 역할을 넘기는 방식이 안전 (아래 B-5).

#### B-5. 각 템플릿의 `def_sram_buffer` 호출부 — 역할 명시
- `mlir_gemm_template.py` 라인 26~28: X→IMEM, W→WMEM, Y→OMEM
- `mlir_conv_template.py` / `mlir_conv_*_template.py`: 동일 패턴 (X/W/Y)
- `mlir_bmm_template.py` 라인 76~78 등: 동일
- `mlir_sdpa_template.py`: query/key/value→IMEM, out→OMEM (K,V를 WMEM로 볼지 설계 결정)
- `prologue_info` (`mlir_gemm_template.py` 라인 235~246 등): `input_dram_var`, `weight_dram_var` 이미 역할 구분됨 → `load_input`(mlir_template.py 863~892)에서 활용 가능.

#### B-6. `mlir_codegen_backend.py` SPAD 용량 체크 (연계: extension_codecache.py)
- 현재 단일 `.spad` 크기로 overflow 판정. 3개 메모리 각각 판정으로 분리 필요(아래 C-3).

---

### [C] 링커 / 주소 레이아웃 (난이도 중)

#### C-1. `PyTorchSimFrontend/extension_codecache.py` link_option (라인 172~175)
- 현재:
  ```python
  link_option = f"-Wl,--section-start=.spad=0x{spad_info['spad_vaddr']:x}"
  ```
- 할 일: 3개 섹션 각각 배치:
  ```python
  link_option = (
    f"-Wl,--section-start=.imem=0x{imem_vaddr:x} "
    f"-Wl,--section-start=.wmem=0x{wmem_vaddr:x} "
    f"-Wl,--section-start=.omem=0x{omem_vaddr:x}"
  )
  ```
- 세 base 주소가 서로 겹치지 않고, 각 용량 안에 들어가도록 배치.

#### C-2. `mlir_codegen_backend.py` `_prepare_simulator_headers` spad_end 심볼 (라인 1147~1150)
- 현재 `.spad`용 end/section_end 심볼 1쌍. → `.imem/.wmem/.omem` 각각 end 심볼 필요.

#### C-3. `extension_codecache.py` SPAD overflow 체크 (라인 200~209)
- 현재:
  ```python
  spad_size = val_llvm_caller.get_spad_size(validation_binary_path)
  spad_usage = stack_size + spad_size
  if CONFIG_SPAD_INFO["spad_size"] < spad_usage: raise SpadOverflowError()
  ```
- 할 일: `.imem/.wmem/.omem` 섹션 크기를 각각 읽어 각 용량과 비교.
  → `get_spad_size`(`mlir_caller_codegen.py`)가 섹션명 인자를 받도록 확장.

#### C-4. `PyTorchSimFrontend/mlir/mlir_caller_codegen.py` `get_spad_size` / `parse_stack_sizes`
- ELF에서 `.spad` 섹션 크기를 읽는 함수. 섹션별로 읽도록 일반화.

---

### [D] Spike (기능 검증) — 난이도 중 (외부 fork 관여)

#### D-1. `Simulator/simulator.py` `run_spike` (라인 137~147)
- 현재 단일 scratchpad 옵션:
  ```python
  spad_option = f"-m...:...,0x{spad_paddr:x}:0x{spad_size*vectorlane_size:x} " \
      f"--scratchpad-base-paddr=... --scratchpad-base-vaddr=... --scratchpad-size=..."
  ```
- **선택지 1 (권장, in-repo만으로 가능)**: IMEM/WMEM/OMEM을 **연속된 하나의 큰 scratchpad**로 매핑.
  - linker에서 3섹션을 `[base, base+총합]` 안에 non-overlap 배치.
  - spike는 여전히 단일 `--scratchpad` 영역으로 커버 → **기능 정확성 OK, fork 수정 불필요**.
- **선택지 2 (정밀)**: spike에 scratchpad 영역 3개 인자 추가 → **riscv-isa-sim fork 수정 필요**.
- 기능검증 단계에선 bank/대역폭이 의미 없으므로 **선택지 1로 충분**.

---

### [E] gem5 (사이클 계산) — **SRAM timing의 핵심**, 난이도 중~상

#### E-1. `gem5_script/script_systolic.py` 메모리 뱅크 구성 (라인 154~177)
- 현재:
  ```python
  spad_num_bank = 1
  system.mem_ranges = [AddrRange(start=0, size="16GB")]
  multi_banked_spm = MultiBankMemorySystem(..., num_banks=spad_num_bank, granule_size=granule_sz)
  system.mem_ctrls = multi_banked_spm.get_ctrls()
  ```
- 할 일: **IMEM/WMEM/OMEM 3개의 주소 영역 + 3개의 `MultiBankMemorySystem`** 생성:
  ```python
  imem = MultiBankMemorySystem(bus, imem_range, num_banks=IMEM_BANKS, total_bandwidth=IMEM_BW)
  wmem = MultiBankMemorySystem(bus, wmem_range, num_banks=WMEM_BANKS, total_bandwidth=WMEM_BW)
  omem = MultiBankMemorySystem(bus, omem_range, num_banks=OMEM_BANKS, total_bandwidth=OMEM_BW)
  system.mem_ctrls = imem.get_ctrls() + wmem.get_ctrls() + omem.get_ctrls()
  ```
- **주소 영역은 linker의 base_vaddr(C-1)와 반드시 일치**해야 함.
- `MultiBankMemorySystem`은 **이 스크립트 안에 이미 구현**되어 있어 확장 쉬움 (bank interleaving 포함).
- bandwidth = `bitwidth × freq`, granule_size = `bitwidth/8`. 메모리별로 다르게 설정.
- **인자 전달**: `script_systolic.py`는 `--vlane`만 받음(라인 20). IMEM/WMEM/OMEM 파라미터를
  넘기려면 `CycleSimulator.compile_and_simulate`(`Simulator/simulator.py` 라인 198~201)에서
  gem5_cmd에 인자 추가 필요.

#### E-2. `Simulator/simulator.py` `CycleSimulator.compile_and_simulate` (라인 198~201)
- gem5 실행 커맨드에 IMEM/WMEM/OMEM bank·대역폭·base주소 인자 추가.

#### E-3. (외부) gem5 `SystolicArray` FU — 3D PE
- `gem5_script/vpu_config.py` 라인 4~9: `systolicArrayWidth/Height` 만 존재(2D 정사각).
- 3D M×N×P PE의 실제 사이클 모델은 **gem5 C++ (PSAL-POSTECH/gem5 fork)** 안.
- SRAM 우선 작업에는 당장 불필요. **3D PE 정밀 모델링은 별도 과제**로 분리 권장.

---

### [F] TOGSim (C++) — 난이도 중 (필요 범위 최소)

#### 현황
- TOGSim은 **DRAM DMA + 타일 스케줄**만 모델링. SRAM 주소/bank/latency는 **모델 안 함**.
- compute 사이클은 gem5에서, DRAM은 ramulator에서 옴.
- `Instruction`(`include/Instruction.h`)은 `dram_addr`만 가짐. SRAM 주소 없음.
- `Tile::_required_sram_size`(`include/Tile.h` 라인 28~30, 54)는 **스케줄 우선순위 정렬용**일 뿐,
  하드 용량/뱅크 제약 아님.

#### F-1. (선택) `include/Tile.h` / `include/scheduler/Scheduler.h`
- IMEM/WMEM/OMEM **개별 용량**을 스케줄러가 추적하려면 `_required_sram_size`를 3분할.
- 현재 정렬 비교(`TileGraph.h` 라인 25, `Scheduler.h` 라인 31)만 영향.
- **SRAM 용량이 스케줄에 실제 영향을 주게 하려면** 이 부분 확장. 아니면 손 안 대도 됨.

#### F-2. `TOGSim/src/TileGraphParser.cc`
- 이미 진행 중인 `systolic_size_n/k` 메타데이터 처리와 동일 계층.
- SRAM bank가 DMA 타이밍에 영향을 주게 하려면 여기서 메모리별 처리 추가 가능(현재는 불필요).

#### 결론: **SRAM "저장 위치 분리"만이 목표라면 TOGSim은 거의 안 건드려도 됨.**
스케줄러가 메모리별 용량을 강제하도록 만들 때만 F-1 필요.

---

### [G] 3D PE Array (M×N×P) — 외부 fork, **별도 과제**

- `pe_array_m/n/k`는 Python(타일링)엔 이미 반영됨.
- 실제 systolic array 사이클 = gem5 `SystolicArray`(C++ fork) + MLIR `memref-to-gemmini` 패스(llvm fork).
- MLIR 패스는 `systolic-array-size={pe_array_n}` **단일값**만 받음
  (`extension_codecache.py` 라인 46~48, 95~99).
- 진정한 N≠K / 3D는 **llvm-project fork + gem5 fork 재빌드** 필요 → SRAM 작업과 분리.

---

## 4. In-repo vs 외부 fork (실현 가능성)

| 작업 | 위치 | repo 내부? | 비고 |
|------|------|:----------:|------|
| Config 추가 (A) | extension_config, yml | ✅ | 쉬움 |
| 버퍼 역할 라우팅 (B) | mlir_codegen_backend, 템플릿 | ✅ | in-repo 핵심 |
| 링커 3섹션 (C) | extension_codecache | ✅ | 주소 배치 주의 |
| Spike 단일영역 커버 (D-선택1) | simulator.py | ✅ | fork 수정 불필요 |
| gem5 뱅크 3분할 (E-1,E-2) | script_systolic.py, simulator.py | ✅ | SRAM timing 핵심, in-repo |
| TOGSim 용량분할 (F, 선택) | TOGSim/*.cc | ✅ | 필요시만 |
| Spike 영역 3개 분리 (D-선택2) | riscv-isa-sim | ❌ fork | 정밀 시 |
| 3D PE 사이클 (G) | gem5 / llvm-project | ❌ fork | 별도 과제 |

→ **SRAM(IMEM/WMEM/OMEM) 분리는 대부분 repo 내부에서 가능.** 외부 fork 없이도 성립.

---

## 5. 추천 작업 순서 (SRAM 우선)

1. **[A] Config**: extension_config 주석 해제 + yml 파라미터 추가 + BaseMLIRHardwareInfo에 info 3개.
2. **[B] 버퍼 라우팅**: `def_sram_buffer`/`allocate_sram_buffer`에 mem_role 추가 → `.imem/.wmem/.omem` 섹션.
   템플릿(gemm→conv→bmm→sdpa 순)에서 역할 지정.
3. **[C] 링커**: 3섹션 base 주소 배치 + overflow 체크 3분할 + get_spad_size 섹션별.
4. **[D] Spike**: 3섹션을 감싸는 단일 scratchpad로 기능검증 통과(선택1).
5. **[E] gem5**: `script_systolic.py`에 3개 뱅크 메모리 시스템 + simulator.py 인자 전달 → SRAM timing 반영.
6. **[F] TOGSim**: (선택) 스케줄러가 메모리별 용량 강제하도록.
7. **[G] 3D PE**: 별도 과제로 분리 (fork 필요).

각 단계마다 **작은 GEMM 예제**로 end-to-end 검증(기능 → 사이클) 후 다음 단계로.

---

6. 반드시 사수님께 확인해야 할 설계 결정

1. **Bias는 어느 메모리에?** (WMEM 합류 vs OMEM vs 별도)
2. **임시/중간 버퍼**(softmax 스크래치, reduction 누산, index_expr 등)는 어느 메모리에?
3. **SDPA의 key/value**를 WMEM(가중치 취급)으로 볼지, IMEM(입력)으로 볼지?
4. **스택(stack)** 은 어느 영역에? (현재 scratchpad 공유) 4번째 영역이 필요한가?
5. **IMEM/WMEM/OMEM 각각의 bank 수 / bitwidth / depth 목표 스펙 값**은?
6. **SRAM 용량 초과를 하드 에러로 강제**할지(스케줄 제약), 소프트 경고만 할지?
7. **3D PE(M×N×P) 정밀 사이클 모델**을 이번에 포함할지, SRAM 먼저 하고 나중에 할지?

---

## 7. 핵심 파일 빠른 참조 (클릭용)

- 설정: `PyTorchSimFrontend/extension_config.py` (77~99), `configs/*.yml`
- HW정보: `PyTorchSimFrontend/mlir/mlir_common.py` (613~625 BaseMLIRHardwareInfo)
- 버퍼할당: `PyTorchSimFrontend/mlir/mlir_codegen_backend.py` (1434~1471, 869, 1147~1153)
- 라우팅 진입점: `PyTorchSimFrontend/mlir/mlir_template.py` (1021~1030 def_sram_buffer, 863~901 load_input/store_output)
- 템플릿: `mlir_gemm_template.py`(26~28,235~246), `mlir_conv*_template.py`, `mlir_bmm_template.py`, `mlir_sdpa_template.py`
- 링커/오버플로: `PyTorchSimFrontend/extension_codecache.py` (172~175, 200~209)
- Spike: `Simulator/simulator.py` (121~167 run_spike)
- gem5: `gem5_script/script_systolic.py` (35~99 MultiBankMemorySystem, 154~177), `gem5_script/vpu_config.py` (4~9)
- gem5 인자: `Simulator/simulator.py` (194~201 CycleSimulator)
- TOGSim: `TOGSim/include/Tile.h` (28~30,54), `TOGSim/include/scheduler/Scheduler.h` (31), `TOGSim/src/TileGraphParser.cc`
