# IMEM/WMEM/OMEM 구현 완료 보고서

**작업 완료 일자**: 2026-07-02  
**상태**: ✅ 기본 구현 완료, ⚠️ 메모리 제약 확인됨

## 요약

SRAM을 단일 .spad 섹션에서 **역할별로 분리된 .imem/.wmem/.omem 섹션**으로 변경하는 구현이 완료되었습니다.

### 구현 범위

**4개 주요 파일 수정:**

1. **extension_config.py** - CONFIG_IMEM_INFO, CONFIG_WMEM_INFO, CONFIG_OMEM_INFO 로드
   - 각 메모리의 vaddr/paddr/size/num_banks 설정
   - Config 파일에 vpu_imem_size_kb_per_lane 키가 있으면 분리 모드 활성화

2. **mlir_common.py** - BaseMLIRHardwareInfo 클래스 수정
   - imem_info/wmem_info/omem_info 초기화
   - `get_mem_target(mem_role)` 메서드로 role→(section, mem_info) 매핑

3. **mlir_codegen_backend.py** - 버퍼 할당 로직
   - `infer_mem_role_from_buffer_name()`: 버퍼 이름(X, W, Y, B)에서 역할 자동 감지
   - `allocate_sram_buffer()`: mem_role 파라미터로 섹션 결정
   - GCC __attribute__((section("..."))) 사용

4. **extension_codecache.py** - 링커 및 시뮬레이터 설정
   - 분리 모드 감지: imem_info/wmem_info/omem_info 모두 not None
   - 링커 옵션: `-Wl,--section-start=.imem=0x... -Wl,--section-start=.wmem=0x... -Wl,--section-start=.omem=0x...`
   - spad_total_size 메커니즘: Spike 물리 메모리 맵 자동 확장

## 테스트 결과

| # | 구성 | 크기 | 상태 | 설명 |
|----|-----|------|------|------|
| 1 | Unified .spad | 32×32 | ✅ Pass | Baseline (역할 구분 없음) |
| 2 | Unified .spad | 128×128 | ✅ Pass | 더 큰 크기도 성공 |
| 3 | IMEM/WMEM/OMEM (32/48/48 KB) | 32×32 | ✅ Pass | 버퍼가 올바른 섹션에 배치됨 |
| 4 | IMEM/WMEM/OMEM (32/48/48 KB) | 128×128 | ❌ Fail | 메모리 부족 (섹션 격리) |
| 5 | IMEM/WMEM/OMEM (384KB 총합) | 32×32 | ❌ Fail | Spike vaddr 범위 제약 (0xD0020000 제한) |
| 6 | Balanced (20/75/33 KB) | 128×128 | ❌ Fail | 메모리 부족 |

### 핵심 발견사항

**✅ 동작 확인:**
```
[ALLOCATE_SRAM] dram_name=X, mem_role=input, section=.imem, ...
[ALLOCATE_SRAM] dram_name=W, mem_role=weight, section=.wmem, ...
[ALLOCATE_SRAM] dram_name=Y, mem_role=output, section=.omem, ...
```

**❌ 제약 조건:**
1. **섹션 격리**: 각 섹션이 완전히 격리되므로 Unified .spad보다 메모리 효율 낮음
2. **Spike vaddr 제약**: --scratchpad-size 파라미터에 vaddr 범위 체크가 고정
   - 128KB 설정 → vaddr [0xD0000000, 0xD0020000) 만 허용
   - spad_total_size로 물리 메모리는 확장하지만 vaddr 체크는 분리 불가능
3. **메모리 사이즈 제약**: 
   - 32×32 matmul: ✅ 안전 (< 10KB per role)
   - 128×128 matmul: ❌ 불가 (role당 > 50KB 필요)

## 사용 방법

### 분리 모드 활성화

Config 파일에 추가:
```yaml
vpu_imem_size_kb_per_lane: 32
vpu_wmem_size_kb_per_lane: 48
vpu_omem_size_kb_per_lane: 48
vpu_imem_num_banks: 8
vpu_wmem_num_banks: 8
vpu_omem_num_banks: 8
```

### Backward Compatibility

키가 없으면 자동으로 unified .spad 모드로 fallback:
```python
if "vpu_imem_size_kb_per_lane" not in config_yaml:
    return None  # None → fallback to .spad
```

## Debug 출력

모든 버퍼 할당이 stderr + /tmp/debug_memory.log에 기록됨:
```
[get_mem_target] role=input -> section=.imem, info keys=[...]
[ALLOCATE_SRAM] dram_name=X, mem_role=input, section=.imem, mem_info type=dict
```

## 생성된 Config 파일

- `systolic_ws_128x128_c1_simple_noc_tpuv3_imem_wmem_omem.yml` - 32/48/48 KB (테스트용)
- `systolic_ws_128x128_c1_simple_noc_tpuv3_imem_wmem_omem_384kb.yml` - 128/128/128 KB (실패 케이스)
- `systolic_ws_128x128_c1_simple_noc_tpuv3_imem_wmem_omem_balanced.yml` - 20/75/33 KB (실패 케이스)

## 다음 단계 (사용자 검토 필요)

1. **메모리 크기 조정**: 사용 케이스에 맞게 IMEM/WMEM/OMEM 크기 결정
2. **타이밍 시뮬레이션**: gem5/TOGSim 타이밍 정확도 검증
3. **Spike 개선**: vaddr 범위 체크를 별도로 구성 가능하게 수정 (384KB 지원)
4. **통합 테스트**: 다른 연산(addmm, linear, conv 등)에서도 정상 동작 확인

## 구현 세부사항

### 메모리 레이아웃 예시 (32/48/48 KB 설정)

```
Virtual Address Space (PE 관점):
0xD0000000 ┌─────────────────────┐
           │   IMEM (32 KB)      │ per lane: 256 bytes × 128 lanes
0xD0008000 ├─────────────────────┤
           │   WMEM (48 KB)      │ per lane: 384 bytes × 128 lanes
0xD0014000 ├─────────────────────┤
           │   OMEM (48 KB)      │ per lane: 384 bytes × 128 lanes
0xD0020000 └─────────────────────┘

Physical Memory (shared by all PEs):
0x2000000000 ┌─────────────────────┐
             │   IMEM (4 MB)       │ 32 KB × 128 lanes
0x2000400000 ├─────────────────────┤
             │   WMEM (6 MB)       │ 48 KB × 128 lanes
0x2000A00000 ├─────────────────────┤
             │   OMEM (6 MB)       │ 48 KB × 128 lanes
0x2001000000 └─────────────────────┘
```

### Linker 명령어
```bash
clang -c test.c \
  -Wl,--section-start=.imem=0xd0000000 \
  -Wl,--section-start=.wmem=0xd0008000 \
  -Wl,--section-start=.omem=0xd0014000 \
  -o test.o
```

### 주요 코드 스니펫

**Config 로드:**
```python
imem_info = extension_config.CONFIG_IMEM_INFO  # None if vpu_imem_size_kb_per_lane missing
wmem_info = extension_config.CONFIG_WMEM_INFO
omem_info = extension_config.CONFIG_OMEM_INFO
```

**역할 감지:**
```python
def infer_mem_role_from_buffer_name(self, dram_name):
    if 'weight' in dram_name.lower() or dram_name.lower() == 'w':
        return "weight"
    elif 'input' in dram_name.lower() or dram_name.lower() == 'x':
        return "input"
    elif 'output' in dram_name.lower() or dram_name.lower() == 'y':
        return "output"
    return None
```

**섹션 배치:**
```c
float buf0_spad[256] __attribute__((section(".imem")));
float buf1_spad[384] __attribute__((section(".wmem")));
float buf2_spad[384] __attribute__((section(".omem")));
```

## 알려진 문제

1. ⚠️ **메모리 격리**: 분리 모드에서 각 role의 크기를 초과하면 다른 role의 메모리를 사용 불가능
   - Unified .spad: 동적 재할당 가능
   - 분리 모드: role별로 고정 크기

2. ⚠️ **Spike vaddr 제약**: 384KB+ 설정은 불가능
   - vaddr 체크가 --scratchpad-size에 고정
   - Spike 에뮬레이터 수정 필요 (사용자 책임)

3. ℹ️ **타이밍 모드 미검증**: functional 모드에서만 테스트됨

---

**작성자**: Claude Code Assistant  
**테스트 플랫폼**: Docker (ghcr.io/psal-postech/torchsim-ci:v1.1.0)  
**실험 날짜**: 2026-07-02 04:30-04:45 UTC
