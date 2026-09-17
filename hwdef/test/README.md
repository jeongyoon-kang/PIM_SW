# hwdef/test — 보드 검증 프로그램

`emulator_top/hwdef/` 의 런타임(`libhwdef.a`)을 써서 보드를 직접 두드리는 도구들.
전부 종료 코드로 판정한다: **0 = pass, 1 = fail, 2 = usage**.

## 빌드

`make` 를 직접 돌리지 말 것. `pim_config.h` 의 컴파일 기본값(ch4)이 들어가서 보드와
주소 체계가 어긋나고, DMA 가 거부되어 프로그램이 즉시 죽는다.

```bash
cd emulator_top
./scripts/setup.sh            # platform/active 를 읽어 그 플랫폼으로 빌드
./scripts/setup.sh --status   # 무엇이 선택됐고 무엇이 빌드됐는지
```

채널 수는 비트스트림의 성질이라 `reprogram.sh --ch N` 이 정하고, setup.sh 는 그걸 읽는다.

## 프로그램

계층 순서대로. 아래로 갈수록 더 많은 것이 이미 동작한다고 가정한다.

| 프로그램 | 무엇을 보나 | 자주 쓰는 인자 |
|---|---|---|
| `emu_sanity` | 보드가 살아 돌아왔나. CFR 쓰고 읽기 | `--no-set` `--quiet` |
| `emu_timing` | DRAM 타이밍 레지스터 8개를 걸고 잰다 | `--scale N` `--no-run` `--keep` `--show` |
| `emu_gpr_loop` | 4 MiB GPR 을 QDMA 로 쓰고 되읽기 | `--mib N` |
| `emu_hbm_direct` | 16개 뱅크 window 가 각각 닿고 서로 구분되나 | `--ch LIST` `--banks LIST` |
| `emu_gemv` | GEMV 타일 하나, 16 lane 전부 검증 | `--ch N` / `--chs 0,1`, `--fill wrvec\|copy\|both`, `--l N` |
| `emu_ewmul` | ISR_EWMUL 그룹 하나, 전 원소 검증 | `--ch N` `--feed/--exec/--dst N` `--l N` `--sweep` |
| `emu_mc` | 메모리 컨트롤러가 호스트와 같게 디코딩하나 + 대역폭 | `--read` / `--write`, `--mib N`, `--at OFF` |
| `emu_chain` | 한 ISR 프로그램 안에서 무엇이 조합되나 | `--test chain\|rewind\|kchunk\|gemv\|all` |

### 자주 걸리는 것

- **`emu_timing` 은 `--keep` 없이는 보드를 원래대로 되돌린다.** scale 을 뒤따르는
  실행까지 남기려면 `--scale N --no-run --keep`.
- **`emu_mc --ch` 는 RoChBaCo 에서 거부된다.** MC 공간 주소는 채널로 나뉘지 않아
  어떤 범위를 잡아도 전 채널에 걸친다. `--ch` 는 ChRoBaCo 전용.
- **`emu_ewmul` 은 `--chs` 가 없다.** 단일 채널만.
- **opsize(`--l`)는 1..64.** `emu_ewmul` 은 `COL+OPSIZE<=64` 라 `--col` 기본값 0 에서만
  64 까지 쓸 수 있다.
- **`emu_gemv --fill`**: `wrvec` = host→GPR→WRVEC→GB (뱅크 트래픽 없음),
  `copy` = host→뱅크 DRAM→COPY→GB (뱅크 read 경유), `both` = 둘 다 돌려 비교(기본값).

## violation 카운터

`emu_gemv` / `emu_ewmul` / `emu_mc` 는 실행 **전에 카운터를 지우고**, 끝나고 표로 찍는다.
그래서 표의 값은 그 실행에서만 발생한 것이다.

```
violation CSR ch0 ANY: 0x0c00
  timing : faw=16 rrd=4 rcd=15 ccd=2 rtp=4 rp=17 wr=28 ras=34
  bank |   RCD_RD    |   CCD_RD    |   RCD_WR    |   CCD_WR    |
       | cnt    max  | cnt    max  | cnt    max  | cnt    max  |
  -----+-------------+-------------+-------------+-------------+
    10 |   1     14  |   0      0  |   0      0  |   0      0  |  <-- ANY
```

- `cnt` — 위반 횟수. 8비트라 255 에서 포화
- `max` — 그 뱅크의 최악 초과량(cycles)
- `timing` — 그 값이 측정된 예산. CFR 에서 직접 읽는다
- `ANY` — 뱅크당 1비트. 표와 대조해 읽는다

**위반은 고장이 아니다.** 에뮬레이터가 "내 메모리가 주어진 예산보다 느렸다"고 보고하는
것이다. 예산(`--scale`)을 키우면 사라진다.

`emu_gemv`/`emu_ewmul` 은 쓰는 채널만, `emu_mc` 는 전 채널을 찍는다.

## 타이밍 레지스터는 커널이 건드리지 않는다

`emu_gemv` 는 예전에 실행 초반 타이밍 8개를 하드코딩 값으로 덮어썼다. 그러면
`emu_timing --scale N --keep` 으로 걸어둔 설정이 되돌려져 타이밍 비교 실험이
성립하지 않는다. 지금은 걸려 있는 값을 읽어 출력만 한다:

```
1. timing (set with ./emu_timing --scale N --keep, this kernel does not write it)
  timing : faw=16 rrd=4 rcd=15 ...
```

## 파형 캡처

`version1.0/` 에 ILA 캡처 결과가 있다. 캡처 방법과 실행 순서는 그쪽 README 참고.

```bash
openila tc1/gemv_ch0        # 파형 열기 (alias, ~/.bashrc)
```
