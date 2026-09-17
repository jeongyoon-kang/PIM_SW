# version1.0 ILA 파형 캡처

DRAM 타이밍 scale 을 바꿔가며 테스트 프로그램을 실행하고, 그 구간의 파형을 ILA 5개에서
cross-trigger 로 동시에 캡처한 것. 이미지는 `emulator_top/hw/ch2/version1.0`.

프로그램 자체에 대한 설명은 상위 [../README.md](../README.md) 참고.

## 실행 순서

각 tc 는 **타이밍을 한 번 걸고, 그 상태로 7종을 차례로 캡처**한다.

```
tc1)  ./emu_timing --scale 1 --no-run --keep
        -> faw=16 rrd=4 rcd=15 ccd=2 rtp=4 rp=17 wr=28 ras=34
      캡처 7종 (아래 표 순서대로)

tc2)  ./emu_timing --scale 5 --no-run --keep
        -> faw=80 rrd=20 rcd=75 ccd=10 rtp=20 rp=85 wr=140 ras=170
      캡처 7종 (동일)
```

`--keep` 이 필수다. 없으면 emu_timing 이 보드를 원래대로 되돌려 뒤따르는 실행에
반영되지 않는다. `--no-run` 은 레지스터만 쓰고 트래픽을 돌리지 않는다.

**캡처 한 건의 내부 순서** (`ila-capture` 스킬의 `capture.py` 가 수행):

```
1. ILA 5개 reset -> depth/window/trigger position 설정
2. axis_ila_0 에 트리거 조건, axis_ila_1~4 는 TRIG_IN_ONLY 체인
3. arm  (하류 4개 -> 소스 순서)
4. 상태를 강제로 읽고 2초 대기          ← arm 이 JTAG 로 반영될 시간
5. 테스트 프로그램 실행, 출력을 <접두사>workload.log 로
6. ILA 5개 각각: 대기 -> upload -> display -> .ila / .csv 저장
7. fix_axi_decode.py 로 AXI 인터페이스 디코딩 속성 보정
```

4번이 없으면 arm 이 반영되기 전에 워크로드가 지나가 트리거가 걸리지 않는다.

## 캡처 7종

| 접두사 | 실행한 명령 | 트리거 |
|---|---|---|
| `gemv_ch0_wrvec_` | `emu_gemv --ch 0 --fill wrvec --l 32` | ISR |
| `gemv_ch0_copy_` | `emu_gemv --ch 0 --fill copy --l 32` | ISR |
| `gemv_chs01_wrvec_` | `emu_gemv --chs 0,1 --fill wrvec --l 32` | ISR |
| `gemv_chs01_copy_` | `emu_gemv --chs 0,1 --fill copy --l 32` | ISR |
| `ewmul_ch0_` | `emu_ewmul --ch 0 --l 32` | ISR |
| `mc_write_` | `emu_mc --write --mib 4` | NoC aw |
| `mc_read_` | `emu_mc --read --mib 4` | NoC ar |

opsize 는 `--l 32` (gemv K=512, ewmul elements=512). emu_gemv 의 ch1 단독은 하지 않는다.
emu_ewmul 은 `--chs` 가 없어 단일 채널만 된다. emu_mc 는 RoChBaCo 라 `--ch` 를 쓸 수 없고
범위가 두 채널에 걸친다.

### 트리거 조건

**ISR (gemv / ewmul)** — `axis_ila_0` 에서 AND
```
emu_top_fpga_i/dispatcher_top_0_m_isr_valid0        = 1
emu_top_fpga_i/emulator_controller_ch0_s_isr_ready  = 1
```
`--chs 0,1` 도 ch0 핸드셰이크를 앵커로 쓴다. multicast 라 어느 쪽이든 5개가 다 잡힌다.

**NoC (emu_mc)** — `axis_ila_0` 에서 OR. emu_mc 는 ISR 을 발행하지 않는다.
```
write:  SLOT_3_AXI_awvalid = 1  OR  SLOT_4_AXI_awvalid = 1
read :  SLOT_3_AXI_arvalid = 1  OR  SLOT_4_AXI_arvalid = 1
```
(SLOT_3/4 = `axi_noc_0_M03/M04_AXI`) read 와 write 는 한 창에 같이 잡히지 않아 따로 뜬다 —
실측으로도 write 캡처엔 aw 만, read 캡처엔 ar 만 나온다.

**체인** — `axis_ila_0` 이 소스(TRIG_OUT 만 있음), `axis_ila_1~3` 이 중계(TRIG_IN+OUT),
`axis_ila_4` 가 종단(TRIG_IN 만). 소스가 트리거되면 5개가 같은 시점에 터진다.

## 캡처 창

window 1개, depth 는 케이스마다 다르다.

| | depth | trigger position | 이유 |
|---|---|---|---|
| tc1 (scale 1) | 1024 | 64 | 동작이 짧아 이 창에 다 들어온다 |
| tc2 (scale 5) | 8192 | 512 | 같은 동작이 사이클 기준 약 5배 늘어나 합성값을 다 쓴다 |

ILA 합성 depth 는 8192 이고 `CONTROL.DATA_DEPTH` 로 런타임에 줄일 수 있다.

## 파일

각 캡처마다 `.ila` 5개 + `.csv` 5개 + `<접두사>workload.log` 1개.

```
<프로그램>_<채널인자>[_<fill>]_axis_ila_{0..4}.{ila,csv}
<프로그램>_<채널인자>[_<fill>]_workload.log
```

`workload.log` 에는 그 실행의 타이밍(`timing : faw=...`), violation 카운터 표, PASS/FAIL,
실행 커맨드 전문이 들어간다. 카운터는 실행 전에 지워지므로 **그 실행에서만 발생한 값**이고,
같은 실행의 파형과 대조할 수 있다.

파형 열기:
```bash
openila tc1/gemv_ch0     # copy + wrvec 10개
openila tc1/ewmul_ch0    #  5개
```

## 결과

70개 캡처 전부 트리거 성공 (tc1 TRIGGER@64, tc2 TRIGGER@512).

`axis_ila_1`(ch0 뱅크 AXI 16슬롯)에서 잰 지연의 중앙값이 `T_RCD` 설정값과 일치한다.

| 실행 | 지표 | tc1 (rcd=15) | tc2 (rcd=75) | 비율 |
|---|---|---|---|---|
| gemv_ch0_wrvec | AR→첫 R beat | 15 | 75 | 5.00x |
| gemv_ch0_copy | AR→첫 R beat | 15 | 75 | 5.00x |
| gemv_chs01_wrvec | AR→첫 R beat | 15 | 75 | 5.00x |
| gemv_chs01_copy | AR→첫 R beat | 15 | 75 | 5.00x |
| ewmul_ch0 | AR→첫 R beat | 15 | 75 | 5.00x |
| mc_read | AR→첫 R beat | 15 | 75 | 5.00x |
| mc_write | AW→첫 W beat | 15 | 75 | 5.00x |

`mc_write` 는 뱅크 read 를 하지 않으므로 AR/R 이 0 이고 AW/W/B 로 재야 한다.
`AW→B` 는 4.14x — 이 구간엔 `T_WR`/`T_RP` 와 버스트 전송 사이클이 섞이고, 전송
사이클은 타이밍 스케일과 무관하게 고정이라 5배보다 낮게 나온다.

## 재현

```bash
./ila/run_version1.0_all.sh
```

레시피는 `ila/recipes/version1.0/`, 캡처 도구는 `.claude/skills/ila-capture/`.
캡처 중에는 Vivado GUI 를 열지 말 것 — 같은 ILA 를 두 세션이 만지면 arm 이 깨진다.
