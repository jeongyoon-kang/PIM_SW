# emu_gemv.gdb — 호스트 쪽 실행 순서를 그대로 찍어보는 gdb 스크립트.
#
#   gdb -q -x emu_gemv.gdb --args ./emu_gemv.dbg --ch 0 --fill wrvec --l 1
#   (gdb) trace_on        # 모든 device touch 를 순서대로 찍고 자동 continue
#   (gdb) stops_on        # 대신 마일스톤에서 멈춰서 직접 들여다보기
#   (gdb) run
#
# 두 모드는 같이 켜도 된다.  trace_on 만 켜면 run 한 번으로 전체 순서 로그가 나온다.
#
# 주의: doorbell 이후의 대기창(2 s)은 벽시계 기준이다.  doorbell 과 poll 사이에서
# 브레이크로 오래 서 있으면 예산을 다 써서 landed=false 로 떨어진다.  그 구간을
# 들여다볼 때는 stops_on 대신 trace_on 을 쓸 것.

set pagination off
set print pretty on
set confirm off

# ---- ISR 한 줄 디코드 -------------------------------------------------------
# emu_isr_get() 이 -O0 빌드에 local symbol 로 남아 있어서 그대로 부른다.
# 비트 필드를 gdb 산술로 다시 구현하면 ROUTE 가 w[1]/w[2] 경계를 넘어가면서
# 헤더와 두 벌이 된다 — 그게 정확히 썩는 자리다.
define isr
  set $p = $arg0
  printf "opcode=0x%02x opsize=%-3d t=%d ch_mask=0x%02x bk=%-2d row=%-6d col=%-2d pu=0x%04x gbmc=0x%04x\n", \
    emu_isr_get($p,59,5), emu_isr_get($p,49,10), emu_isr_get($p,35,1), \
    emu_isr_get($p,27,8), emu_isr_get($p,23,4), emu_isr_get($p,6,17), \
    emu_isr_get($p,0,6), emu_isr_get($p,78,16), emu_isr_get($p,176,16)
  printf "    route:"
  set $i = 0
  while $i < 16
    printf " %d", emu_isr_get($p, 96 + 5*$i, 5)
    set $i = $i + 1
  end
  printf "   (16=GB, 31=unused)\n"
  printf "    raw  : %016llx%016llx%016llx%016llx\n", $p->w[3], $p->w[2], $p->w[1], $p->w[0]
end
document isr
isr &prog_wrvec[1]  — ISR 한 개를 필드로 풀어서 찍는다
end

define isrdump
  set $n = $arg1
  set $j = 0
  while $j < $n
    printf "  [%u] ", $j
    isr &$arg0[$j]
    set $j = $j + 1
  end
end
document isrdump
isrdump prog_wrvec NPROG  — 프로그램 전체를 순서대로 찍는다
end

# ---- 순서 추적 --------------------------------------------------------------
define trace_on
  set $seq = 0

  # ISR 조립 순서.  아직 보드는 건드리지 않는다 — 메모리에 만들기만 한다.
  break emu_isr_build
  commands
    silent
    set $seq = $seq + 1
    printf "[%02d] BUILD   opcode=0x%02x opsize=%d row=%d col=%d ch_mask=0x%02x pu=0x%04x gbmc=0x%04x  -> %s\n", \
      $seq, s->opcode, s->opsize, s->row, s->col, s->ch_mask, s->pu_mask, s->gb_mc_mask, "(host memory only)"
    continue
  end

  # HBM 직접 구멍.  GPR/IMEM 도 axi_write 로 내려가므로 주소로 갈라 찍는다.
  break axi_write
  commands
    silent
    set $seq = $seq + 1
    if axi >= 0x4000000000 && axi < 0x20200000000
      printf "[%02d] HBM  W  axi=0x%011llx len=%-6zu  (direct aperture: 피연산자/벡터 staging)\n", $seq, axi, len
    else
      printf "[%02d]  ...     axi=0x%011llx len=%zu\n", $seq, axi, len
    end
    continue
  end

  break gpr_write
  commands
    silent
    set $seq = $seq + 1
    printf "[%02d] GPR  W  word=%-5u n=%-3u  (H2C)\n", $seq, word, n
    continue
  end

  break gpr_read
  commands
    silent
    set $seq = $seq + 1
    printf "[%02d] GPR  R  word=%-5u n=%-3u  (C2H)\n", $seq, word, n
    continue
  end

  break imem_write
  commands
    silent
    set $seq = $seq + 1
    printf "[%02d] IMEM W  word=%-5u n=%-3u  <== 여기가 dispatcher IMEM 다운로드\n", $seq, word, n
    continue
  end

  # CFR 쓰기만 건다.  cfr_rd 는 require_idle/poll 루프에서 수천 번 돈다.
  break cfr_wr
  commands
    silent
    set $seq = $seq + 1
    if off == 0x28
      printf "[%02d] CFR  W  PROG_LEN = %u\n", $seq, v
    end
    if off == 0x00
      printf "[%02d] CFR  W  CTRL     = 0x%x   <== DOORBELL (되돌릴 수 없는 지점)\n", $seq, v
    end
    if off != 0x28 && off != 0x00
      printf "[%02d] CFR  W  off=0x%02x     = 0x%x\n", $seq, off, v
    end
    continue
  end

  break require_idle
  commands
    silent
    set $seq = $seq + 1
    printf "[%02d] IDLE?   before: %s\n", $seq, when
    continue
  end

  break viol_clear_all
  commands
    silent
    set $seq = $seq + 1
    printf "[%02d] VIOL    clear all channels\n", $seq
    continue
  end

  break run_program
  commands
    silent
    set $seq = $seq + 1
    printf "\n[%02d] ===== run_program(n=%u, dst_word=%u, ndst=%u) =====\n", $seq, n, dst_word, ndst
    continue
  end
  printf "trace_on: 모든 device touch 에 브레이크를 걸었다.  run 하면 순서대로 찍힌다.\n"
end

# ---- 마일스톤에서 실제로 멈추기 ---------------------------------------------
define stops_on
  break emu_gemv.c:671
  break emu_gemv.c:700
  break emu_gemv.c:713
  break emu_gemv.c:724
  break emu_gemv.c:744
  break emu_gemv.c:344
  break emu_gemv.c:349
  break emu_gemv.c:356
  break emu_gemv.c:364
  break emu_gemv.c:377
  printf "stops_on: 마일스톤 브레이크.  info breakpoints 로 확인.\n"
  printf "  671 require_idle #0   700 HBM 피연산자   713 GPR 벡터(WRVEC)\n"
  printf "  724 DRAM 벡터(COPY)   744 run_program    344 IMEM write\n"
  printf "  349 poison            356 PROG_LEN       364 DOORBELL   377 결과 GPR read\n"
end
