# riscv-pk elf.c brk_min fix (SRAM/tensor-heap overlap)

컨테이너 `/workspace/riscv-pk/pk/elf.c`의 brk_min 계산 루프에 조건 추가:
`.imem/.wmem/.omem`(0xd0000000+) SRAM 세그먼트를 brk_min에서 제외해서
텐서 heap이 SRAM lane 공간과 안 겹치게 함.

```c
if (vaddr < 0xd0000000UL && vaddr + ph[i].p_memsz > info->brk_min)
    info->brk_min = vaddr + ph[i].p_memsz;
```

재빌드: `cd /workspace/riscv-pk/build && make pk`
패치본 전체: `patches/riscv-pk-elf.c.brk-fix`
