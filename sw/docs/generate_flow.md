# generate.py 실행 흐름

> `sw/app/generate.py` 를 실행했을 때 코드가 어떤 순서로 도는지, 파일과 줄 위치를
> 따라가며 정리한다. Python 진입점에서 시작해 pybind 바인딩(`_pim.cpp`), C 런타임
> (`runtime/`), 그리고 doorbell 까지 내려간다.
>
> 줄 번호는 2026-09-29 코드 기준이다. 코드가 바뀌면 줄이 밀릴 수 있으니, 링크가
> 어긋나면 함수 이름으로 찾는다.

기준으로 삼은 실행은 기본 옵션이다.

```
./generate.py            # --model llama-3.2-1b --backend pim --max-new-tokens 4096
```

---

## 0. 한눈에 보기

```
generate.py main()
 ├─ 1. 보드 구조 읽기        device.geometry()  ──ctypes──▶ libpim pim_geom()
 ├─ 2. 배치 계산             budget.plan()      (계산만, 보드 사용 없음)
 ├─ 3. 모델 적재             PimModel.__init__
 │     ├─ ops.Runtime        ──pybind──▶ pim_rt_open()     GPR 확보, 엔진 열기
 │     ├─ attention.register("pim")
 │     ├─ HF from_pretrained (가중치가 호스트에 올라옴)
 │     ├─ _swap_linears      nn.Linear → PimLinear         가중치 업로드
 │     └─ PimCache           층마다 빈 PimLayer
 ├─ 4. 생성                  PimModel.generate → HF model.generate
 │     └─ forward 마다: PimLinear / PimLayer.update / pim_attention_forward
 │            └─ pim_op_begin → pim_op_add → pim_op_submit
 │                   └─ lower → pim_exec_run (IMEM 적재, doorbell, 폴링) → 결과 읽기
 └─ 5. 통계 출력, 카드 메모리 해제
```

우리 코드가 HF transformers 에 끼어드는 곳은 세 군데다.

1. `nn.Linear` 를 `PimLinear` 로 바꾼다 — 모든 선형층과 lm_head 가 카드에서 돈다.
2. `past_key_values` 로 `PimCache` 를 넘긴다 — K/V 가 카드에 쌓인다.
3. `"pim"` 어텐션 함수를 등록한다 — q·Kᵀ 와 s·V 가 카드에서 돈다.

생성 루프, 마스크 크기 계산, RoPE, RMSNorm, SiLU, residual, 임베딩 조회, argmax 는
transformers 의 원래 코드가 호스트에서 처리한다.

---

## 1. 진입과 인자

- [generate.py:231-232](../app/generate.py#L231-L232) — `main()` 호출.
- [generate.py:26-29](../app/generate.py#L26-L29) — 스크립트 폴더를 `sys.path` 에 넣고
  `pimllm.budget`, `pimllm.device` 를 import 한다. torch 와 `_pim` 은 3단계에서 처음
  불러온다.
- [generate.py:96-132](../app/generate.py#L96-L132) — 인자 파싱.

## 2. 보드 구조(geometry) 읽기

- [generate.py:135-142](../app/generate.py#L135-L142) — `--ch` 가 없으면 `geometry()` 호출.
  - [device.py:166](../app/pimllm/device.py#L166) → [device.py:122 `_load()`](../app/pimllm/device.py#L122)
    — ctypes 로 `sw/lib/libpim.so` 를 연다.
  - [device.py:171](../app/pimllm/device.py#L171) — C 함수 `pim_geom()` 으로 채널 수,
    뱅크 수, 행 크기, DRAM·GPR 영역을 받는다.
  - [device.py:204](../app/pimllm/device.py#L204) — ctypes 구조체 정의가 C 헤더와
    맞는지 `unit_bytes = nch × nbank × row_bytes` 로 확인한다.
- [generate.py:144-149](../app/generate.py#L144-L149) — 구조를 출력하고,
  [device.py:213 `meminfo()`](../app/pimllm/device.py#L213) 로 지금 쓰고 있는 카드
  메모리를 보여준다.

보드가 없을 때의 흐름은 [7절](#7-보드가-없을-때-pimko-미적재) 에 따로 정리했다.

## 3. 배치 계산(budget)

- [generate.py:152](../app/generate.py#L152) → [generate.py:54 `resolve_shape`](../app/generate.py#L54)
  — `llama-3.2-1b` 같은 이름은 [budget.py:281 `KNOWN`](../app/pimllm/budget.py#L281) 의
  모양을 쓰고, 그 밖의 이름은 HF `AutoConfig` 로 받아온다.
- [generate.py:157](../app/generate.py#L157) → [budget.py:239 `plan()`](../app/pimllm/budget.py#L239)
  — 층마다 q/k/v/o/gate/up/down 7개 행렬과 lm_head 의 카드 바이트 수를
  [budget.py:31 `tensor_bytes`](../app/pimllm/budget.py#L31) 로 계산한다.
- [generate.py:160-165](../app/generate.py#L160-L165) — `--dry-run` 이면 여기서 끝난다.
  가중치가 카드에 들어가지 않으면 실행을 거부한다.

## 4. 모델 적재 — `PimModel.__init__`

[generate.py:177-186](../app/generate.py#L177-L186) 에서 `PimModel(...)` 을 만든다.

1. [model.py:179-180](../app/pimllm/model.py#L179-L180) — HF config 로 `ModelShape` 를 만든다.
2. [model.py:187-198](../app/pimllm/model.py#L187-L198) — `plan()` 을 한 번 더 계산하고
   최대 토큰 수를 정한다. `verbose=True` 라서 배치 보고서가 두 번째로 출력된다.
3. [model.py:219-226](../app/pimllm/model.py#L219-L226) — `ops.Runtime(...)` 생성.
   - [ops.py:204](../app/pimllm/ops.py#L204) → [_pim.cpp:140-151](../app/_pim.cpp#L140-L151)
     → [pim_op.c:144 `pim_rt_open`](../runtime/pim_op.c#L144).
   - `_pim` 모듈은 [ops.py:34](../app/pimllm/ops.py#L34) 에서 이때 처음 import 된다.
4. [model.py:228](../app/pimllm/model.py#L228) → [attention.py:148 `register`](../app/pimllm/attention.py#L148)
   — `"pim"` 어텐션 함수를 등록한다.
5. [model.py:231-234](../app/pimllm/model.py#L231-L234) — tokenizer 와 모델을 불러온다
   (`attn_implementation="pim"`, bf16). 이 시점에는 가중치가 호스트 메모리에 있다.
6. [model.py:238](../app/pimllm/model.py#L238) → [model.py:139 `_swap_linears`](../app/pimllm/model.py#L139)
   — 모든 `nn.Linear` 를 하나씩 `PimLinear` 로 바꾼다.
   - [model.py:151](../app/pimllm/model.py#L151) → [model.py:108](../app/pimllm/model.py#L108)
     `ops.Tensor.from_weight` → [ops.py:124-131](../app/pimllm/ops.py#L124-L131)
     → [_pim.cpp:65-69](../app/_pim.cpp#L65-L69) `pim_tensor_alloc`
     ([pim_tensor.c:309](../runtime/pim_tensor.c#L309)) → `pim_tensor_upload`
     ([pim_tensor.c:429](../runtime/pim_tensor.c#L429)).
   - [model.py:152-157](../app/pimllm/model.py#L152-L157) — 부모 모듈을 통해 교체하고,
     원래 Linear 를 바로 지운 뒤 `gc.collect()` 로 호스트 메모리를 돌려준다.
7. [model.py:243](../app/pimllm/model.py#L243) → [cache.py:219 `PimCache`](../app/pimllm/cache.py#L219)
   — 층 수만큼 `PimLayer` 를 만든다.
8. [model.py:247-250](../app/pimllm/model.py#L247-L250) — `--isa-trace` 를 준 경우 forward
   직전 훅을 단다.

## 5. 생성 — `PimModel.generate`

- [generate.py:199](../app/generate.py#L199) → [model.py:260](../app/pimllm/model.py#L260)
- [model.py:273-285](../app/pimllm/model.py#L273-L285) — `--chat` 이면 채팅 템플릿을
  씌우고, 아니면 프롬프트를 그대로 토크나이즈한다.
- [model.py:287](../app/pimllm/model.py#L287) — 프롬프트 + 새 토큰 수가 `max_tokens` 를
  넘으면 NOTE 를 출력한다.
- [model.py:291](../app/pimllm/model.py#L291) → [cache.py:164 `reset`](../app/pimllm/cache.py#L164)
  — KV 위치를 0 으로 돌린다.
- [model.py:292](../app/pimllm/model.py#L292) — `TimedStreamer` 생성.
- [model.py:295](../app/pimllm/model.py#L295) — HF `model.generate(...)` 에 제어가 넘어간다.
  greedy 루프는 transformers 가 돌리고, 우리 코드는 아래 5.1 의 세 곳에서 불린다.

### 5.1 forward 한 번 (층마다 반복)

첫 forward(prefill)는 S = 프롬프트 길이, 이후 forward(decode)는 S = 1 이다. HF
`LlamaDecoderLayer` 가 부르는 순서대로 적는다.

1. **RMSNorm** — 호스트.
2. **q_proj / k_proj / v_proj** — [model.py:113 `PimLinear.forward`](../app/pimllm/model.py#L113).
   [model.py:123-129](../app/pimllm/model.py#L123-L129) 에서 위치(행)마다 `rt.matvec` 를
   한 번씩 부른다. 호출 경로는 [5.2](#52-matvec-한-번이-카드까지-가는-길) 참고.
3. **RoPE** — 호스트.
4. **KV 저장** — HF 가 `past_key_values.update(...)` 를 부른다 →
   [cache.py:114 `PimLayer.update`](../app/pimllm/cache.py#L114).
   - 첫 호출이면 [cache.py:91 `lazy_initialization`](../app/pimllm/cache.py#L91) — 들어온
     텐서 모양에서 KV 헤드 수와 head_dim 을 읽고, K 는 OUT_PACKED, V 는 RED_MAJOR 로
     growable 텐서를 만든다.
   - [cache.py:130-131](../app/pimllm/cache.py#L130-L131) `_grow` — 필요한 만큼 페이지를
     할당한다 → [pim_tensor.c:363 `pim_tensor_grow`](../runtime/pim_tensor.c#L363).
   - [cache.py:132-133](../app/pimllm/cache.py#L132-L133) `append` →
     [pim_tensor.c:479 `pim_tensor_append`](../runtime/pim_tensor.c#L479).
   - [cache.py:135](../app/pimllm/cache.py#L135) — 텐서 대신 층 객체 자신을 돌려준다.
     이 객체가 그대로 어텐션 함수의 `key`, `value` 인자가 된다.
5. **어텐션** — HF 가 등록된 `"pim"` 함수를 부른다 →
   [attention.py:38 `pim_attention_forward`](../app/pimllm/attention.py#L38).
   - [attention.py:100-111](../app/pimllm/attention.py#L100-L111) — **q·Kᵀ**. 층 하나의
     모든 위치·모든 헤드를 `rt.batch()` 하나에 `add_heads` 로 쌓는다 →
     [ops.py:324](../app/pimllm/ops.py#L324) → [pim_op.c:580 `pim_op_add_heads`](../runtime/pim_op.c#L580).
     `with` 블록이 끝날 때 [ops.py:352-355](../app/pimllm/ops.py#L352-L355) 에서 submit 한다.
   - [attention.py:118-127](../app/pimllm/attention.py#L118-L127) — **causal mask + softmax**,
     호스트에서 fp32 로 한 번에 계산한다.
   - [attention.py:136-141](../app/pimllm/attention.py#L136-L141) — **s·V**. 헤드마다
     `bat.add` 로 쌓아 batch 하나로 submit 한다.
   - [attention.py:145](../app/pimllm/attention.py#L145) — `[1, S, H_q, D]` 로 돌려준다.
6. **o_proj** — `PimLinear`. 이어서 residual 을 더한다(호스트).
7. **RMSNorm → gate_proj, up_proj (`PimLinear`) → SiLU·곱(호스트) → down_proj (`PimLinear`)**,
   이어서 residual.

모든 층이 끝나면 마지막 norm 을 거쳐 **lm_head**(`PimLinear`)를 돈다. lm_head 는 출력
그룹이 많아서 5.2 의 "중간 submit" 경로로 여러 doorbell 에 나뉜다. 그 뒤 HF 가
argmax 로 다음 토큰을 고른다(`do_sample=False`).

### 5.2 matvec 한 번이 카드까지 가는 길

[ops.py:217 `Runtime.matvec`](../app/pimllm/ops.py#L217) → [_pim.cpp:201](../app/_pim.cpp#L201)
→ [pim_op.c:663 `pim_op_matvec`](../runtime/pim_op.c#L663) 은 begin + add + submit 을
차례로 부른다. batch 경로(`rt.batch()`)는 같은 함수들을 쓰되 add 를 여러 번 한 뒤
submit 을 한 번 한다.

| 단계 | 위치 | 하는 일 |
|---|---|---|
| begin | [pim_op.c:328](../runtime/pim_op.c#L328) | 논리 프로그램을 비우고 GPR 사용량을 0 으로 돌린다 |
| add | [pim_op.c:623](../runtime/pim_op.c#L623) | 한 launch 에 들어갈 만큼씩 출력 그룹을 나눠 `add_one` 을 부른다 |
| add_one | [pim_op.c:420](../runtime/pim_op.c#L420) | 조각 하나를 현재 프로그램에 붙인다 |
| └ 자리 확인 | [pim_op.c:461-472](../runtime/pim_op.c#L461-L472) | 조각 수·벡터 GPR·결과 GPR·명령어 배열·IMEM 중 하나라도 넘치면 지금까지를 submit 하고 새로 begin 한다 |
| └ 벡터 업로드 | [pim_op.c:484-490](../runtime/pim_op.c#L484-L490) | beat 경계까지 0 으로 채워 GPR 로 보낸다 |
| └ 명령 생성 | [pim_op.c:493](../runtime/pim_op.c#L493) | `pim_matvec_logical_part` ([pim_matvec.c:167](../runtime/pim_matvec.c#L167)) 가 WRVEC / MAC / RD_MAC 논리 명령을 만든다 |
| submit | [pim_op.c:350](../runtime/pim_op.c#L350) | 아래 네 단계 |
| └ EOS | [pim_op.c:364](../runtime/pim_op.c#L364) | 프로그램 끝 표시 |
| └ lower | [pim_op.c:367](../runtime/pim_op.c#L367) | `pim_prog_lower_ctx` ([pim_lower.c:202](../runtime/pim_lower.c#L202)) 가 물리 ISR 로 바꾼다 |
| └ 실행 | [pim_op.c:371](../runtime/pim_op.c#L371) → [pim_exec.c:453](../runtime/pim_exec.c#L453) | IMEM 적재 → PROG_LEN 기록 → doorbell → STATUS.DONE 폴링 → RUN_CYC 읽기 |
| └ 결과 | [pim_op.c:388-411](../runtime/pim_op.c#L388-L411) | 결과 GPR 을 한 번에 읽어와 (그룹, 채널, 뱅크) 순서를 출력 순서로 다시 놓는다 |

### 5.3 토큰마다

- HF 가 새 토큰을 streamer 에 넘긴다 → [model.py:56 `on_finalized_text`](../app/pimllm/model.py#L56)
  — 토큰, 걸린 시간, 그 토큰에 쓴 ISA 수를 한 줄로 출력한다.
- EOS 가 나오거나 `max_new_tokens` 에 닿을 때까지 5.1 을 S = 1 로 반복한다.
- 카드 메모리가 다 차면 `_grow` 가 `PimOutOfMemory` 를 던지고
  [model.py:299-301](../app/pimllm/model.py#L299-L301) 에서 받아 멈춘다.
- 끝나면 [model.py:69-81](../app/pimllm/model.py#L69-L81) 에서 전체 텍스트와 tok/s 를 출력한다.

## 6. 마무리

- [generate.py:201-204](../app/generate.py#L201-L204) — `rt.stats()` 로 op 수, launch 수,
  ISA 수, RUN_CYC 를 출력한다.
- [generate.py:207](../app/generate.py#L207) → [model.py:307 `free`](../app/pimllm/model.py#L307)
  — KV 캐시와 모든 `PimLinear` 가중치의 카드 메모리를 해제한다.

---

## 7. 보드가 없을 때 (pim.ko 미적재)

가상 2채널 구조로 끝까지 가는 것은 `--dry-run` 뿐이다. 실제 생성은 Runtime 을 만들
때 멈춘다.

| # | 위치 | 일어나는 일 |
|---|---|---|
| 1 | [generate.py:139](../app/generate.py#L139) → [device.py:170-171](../app/pimllm/device.py#L170-L171) | `libpim.so` 는 로드된다. `pim_geom()` → [pim_dev.c:368](../lib/pim_dev.c#L368) → `pim_default()` 가 `/dev/pim` 열기에 실패해 NULL 을 돌려준다 |
| 2 | [device.py:172-176](../app/pimllm/device.py#L172-L176) | `PimUnavailable` |
| 3 | [generate.py:140-142](../app/generate.py#L140-L142) | 예외를 받아 `no device: ...` 를 출력하고, 가정한 2채널 구조([generate.py:32](../app/generate.py#L32))로 바꿔 끼운다 |
| 4 | [generate.py:146](../app/generate.py#L146) | `meminfo()` 는 건너뛴다 |
| 5 | [generate.py:157-158](../app/generate.py#L157-L158) | 배치 보고서 출력(가상 2채널 기준) |
| 6 | [generate.py:160-161](../app/generate.py#L160-L161) | `--dry-run` 이면 여기서 정상 종료 |
| 7 | [generate.py:178](../app/generate.py#L178) | `pimllm.model` import. `import _pim` 도 성공한다 — 모듈 초기화([_pim.cpp:285-294](../app/_pim.cpp#L285-L294))는 상수만 등록한다 |
| 8 | [model.py:179](../app/pimllm/model.py#L179) | HF config 를 받는다(HF 캐시나 네트워크 필요) |
| 9 | [model.py:187-189](../app/pimllm/model.py#L187-L189) | 배치 보고서를 한 번 더 출력 |
| 10 | [model.py:219](../app/pimllm/model.py#L219) → [_pim.cpp:150](../app/_pim.cpp#L150) | `pim_rt_open(ctx(), ...)` 의 인자를 계산하다 [_pim.cpp:46-53 `ctx()`](../app/_pim.cpp#L46-L53) 가 `RuntimeError: cannot open the PIM device` 를 던진다 |
| 11 | — | `generate.py` 는 이 예외를 받지 않으므로 traceback 과 함께 종료 코드 1 로 끝난다 |

10번이 [model.py:232](../app/pimllm/model.py#L232) 의 `from_pretrained` 보다 앞에 있으므로
가중치는 내려받지 않는다. `--backend torch` 는 보드를 쓰지 않으니 CPU 로 끝까지 돈다.

보드를 여는 경로는 두 갈래다. `device.py` 는 ctypes 로 `libpim` 을 부르고, `_pim` 은
pybind 로 같은 `libpim` 의 `pim_default()` 를 부른다. `pim_default()` 는 첫 호출 때 한
번만 장치를 열고 그 결과를 보관하므로([pim_dev.c:342](../lib/pim_dev.c#L342)), 두 경로의
판정은 항상 같다.

---

## 8. 클래스 초기화에서 볼 점

카드 메모리를 잡는 곳은 세 군데다.

| 무엇 | 어디서 | 언제 |
|---|---|---|
| GPR 3.5 MiB (벡터 1 MiB + 결과 2.5 MiB) | `pim_rt_open` | Runtime 생성 때 한 번 |
| 가중치 DRAM | `PimLinear.__init__` | 모델 적재 때 전부 |
| KV DRAM | `PimLayer._grow` | 토큰이 늘어 페이지 경계를 넘을 때마다 |

### `PimModel.__init__` — [model.py:171-250](../app/pimllm/model.py#L171-L250)

호출 순서에 의존 관계가 있다.

| 순서 | 위치 | 이 자리에 있는 이유 |
|---|---|---|
| budget | [:187](../app/pimllm/model.py#L187) | 모델을 내려받기 전에 들어가는지 판단한다 |
| Runtime | [:219](../app/pimllm/model.py#L219) | `PimLinear` 가 `rt` 를 들고 있어야 하므로 swap 앞에 온다 |
| `attention.register` | [:228](../app/pimllm/model.py#L228) | `from_pretrained` 가 `"pim"` 이름을 검사하므로 그 앞에 온다 |
| swap → cache | [:238](../app/pimllm/model.py#L238), [:243](../app/pimllm/model.py#L243) | 가중치가 먼저 자리를 잡고 KV 는 남은 공간을 쓴다 |

- Runtime 인자 네 개([:219-226](../app/pimllm/model.py#L219-L226))가 launch 크기의 틀을
  정한다. `max_red` 는 가장 긴 벡터(FFN intermediate), `max_out_groups` 는 lm_head 의
  vocab 을 그룹 수로 바꾼 값, `vec_bytes` / `res_bytes` 는 GPR 4 MiB 를 벡터용과
  결과용으로 나눈 몫이다. batch 하나에 조각이 몇 개 들어가는지는 주로 이 두 값이
  정한다.
- 예산은 인자로 받은 `geometry` 로 계산하고, Runtime 과 PimCache 는 `_pim.geometry()`
  로 실제 보드를 읽는다([ops.py:212](../app/pimllm/ops.py#L212), [cache.py:221](../app/pimllm/cache.py#L221)).
  `--ch N` 을 실제 보드와 다르게 주면 예산은 N 채널, 실행은 실제 채널 기준이 된다.
  `--ch` 는 dry-run 용으로 쓰는 것이 맞다.
- `max_tokens` ([:198](../app/pimllm/model.py#L198))는 `generate` 에서 NOTE 출력에 쓰인다.
  실제로 멈추는 지점은 KV `_grow` 가 실패할 때다.

### `ops.Runtime.__init__` → `pim_rt_open` — [ops.py:190-215](../app/pimllm/ops.py#L190-L215), [pim_op.c:144-265](../runtime/pim_op.c#L144-L265)

- 실행 엔진 열기([pim_op.c:167-171](../runtime/pim_op.c#L167-L171)) — `pim_exec_open` 이
  DRAM 타이밍 레지스터를 읽고 T_CCD 가 2 보다 작으면 거부한다. 보드 상태에서 오는
  에러는 대개 여기서 처음 보인다.
- `max_launch_groups` ([pim_op.c:177-197](../runtime/pim_op.c#L177-L197)) — 가장 긴 reduction
  기준으로 IMEM 명령어 한도 안에 supergroup 이 몇 개 들어가는지 센다. lower 뒤의
  명령어 수로 센다(drain 하나가 채널 수만큼의 ISR 이 된다). lm_head 가 doorbell 몇
  번으로 나뉘는지가 여기서 나온다.
- GPR 두 덩어리([pim_op.c:199-213](../runtime/pim_op.c#L199-L213)) — `vgpr`, `ygpr` 를 한 번
  잡고, batch 안에서는 앞에서부터 잘라 쓴다. 5.2 의 "자리 확인" 조건 가운데 둘이 이
  크기에서 온다.
- 명령어 배열([pim_op.c:222-231](../runtime/pim_op.c#L222-L231)) — IMEM 한도 크기로 잡는다.
  `max_batch` (4096)는 조각 메타데이터 배열의 크기다. launch 하나에 들어가는 실제 조각
  수는 IMEM 과 GPR 이 정한다.
- `dual_latch` → `allow_t_latch` — 이 값이 꺼져 있으면 [pim_op.c:333](../runtime/pim_op.c#L333)
  에서 DUAL 모드를 거부한다.
- Python 쪽에 보관하는 `per_group`, `row_elems` ([ops.py:213-215](../app/pimllm/ops.py#L213-L215))
  는 attention 이 헤드를 DRAM 행에 나눠 담을 때 쓴다.

### `PimLinear.__init__` — [model.py:92-111](../app/pimllm/model.py#L92-L111)

- [:108](../app/pimllm/model.py#L108) 한 줄이 핵심이다. 카드 메모리를 할당하고 가중치를
  PCIe 로 올린다. 모델 적재 시간의 대부분이 여기다.
- 모듈에는 `self.weight` 가 없고 `bias` 만 호스트에 둔다([:111](../app/pimllm/model.py#L111)).
  그래서 swap 직후 원래 Linear 를 지우면 호스트 메모리가 바로 풀린다.
- `node`, `layer` ([:103-106](../app/pimllm/model.py#L103-L106))는 ISA trace 라벨, `_span` 은
  profile 에서 종류별로 묶어 보는 이름이다.

### `ops.Tensor.__init__` / `growable` — [ops.py:106-116](../app/pimllm/ops.py#L106-L116)

- 일반 생성자(가중치)는 만들 때 전체 크기를 할당한다.
- `growable` (KV)은 [_pim.cpp:73-79](../app/_pim.cpp#L73-L79) 의
  `pim_tensor_plan_growable` 이 레이아웃 계획만 세우고, 할당은 `grow` 때 한다.
- `layout` (OUT_MAJOR / OUT_PACKED / RED_MAJOR)이 `noutpad`, `nredpad`, `pack` 을 정하고,
  이 값들이 [pim_op.c:432](../runtime/pim_op.c#L432) 의 범위 검사와
  [pim_op.c:640](../runtime/pim_op.c#L640) 의 `out_first % pack` 검사에 쓰인다.

### `PimCache.__init__` / `PimLayer.__init__` — [cache.py:219-223](../app/pimllm/cache.py#L219-L223), [cache.py:77-88](../app/pimllm/cache.py#L77-L88)

- 빈 틀만 만든다. 모든 층이 `events` 리스트 하나를 같이 써서, KV 증가 기록이 전 층의
  할당 순서를 한 줄로 담는다.
- 실제 초기화는 첫 `update` 때의 [`lazy_initialization`](../app/pimllm/cache.py#L91) 이다.
  설정값이 아니라 들어온 텐서 모양으로 헤드 수와 head_dim 을 정한다.
- 층을 처음부터 전부 만드는 이유 — HF 의 `layer_class_to_replicate` 는 인자 없는
  생성자를 요구해서 `rt` 를 넘길 수 없다([cache.py:213](../app/pimllm/cache.py#L213)).

### 그 밖

- `TimedStreamer.__init__` ([model.py:48-54](../app/pimllm/model.py#L48-L54)) — `t0` 는 streamer
  생성 시점부터 잰다. 첫 토큰 줄의 시간과 ISA 수에는 prefill 전체가 들어간다.
- `Batch.__init__` ([ops.py:298-300](../app/pimllm/ops.py#L298-L300)) — 실제 동작은
  `__enter__` (= `pim_op_begin`)와 `__exit__` (= `pim_op_submit`)에 있다. `_keep` 은 submit
  이 결과를 써 넣을 때까지 출력 버퍼를 살려 둔다.

---

## 9. 읽을 때 알아둘 점

- **doorbell 횟수.** 선형층은 위치마다 `matvec` 을 불러 prefill 에서 위치 수만큼
  doorbell 이 울린다. 어텐션은 층당 q·Kᵀ batch 하나, s·V batch 하나로 묶는다.
  prefill 이 느리면 먼저 볼 곳은 [model.py:123](../app/pimllm/model.py#L123) 의 행 단위
  루프다.
- **배치 보고서가 두 번 나온다.** [generate.py:157](../app/generate.py#L157) 과
  [model.py:187](../app/pimllm/model.py#L187) 에서 각각 `plan()` 을 부르기 때문이다.
- **다른 옵션.** `--profile` / `--trace` 는 [generate.py:190-197](../app/generate.py#L190-L197)
  에서 [profile.py](../app/pimllm/profile.py) 의 `Profile` 로 `generate` 를 감싼다.
  `--backend torch` 는 [generate.py:211-228](../app/generate.py#L211-L228) 의 CPU 참조
  경로로 간다.
