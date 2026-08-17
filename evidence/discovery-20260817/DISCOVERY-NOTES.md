# Paper-2 Discovery Probes — ALL FOUR DOORS (2026-08-17)

Handoff §5 step 1 executed. Doors: **1 = Bedrock Converse** (`us.openai.gpt-5.6-sol`, profile default, us-east-1) · **2 = Bedrock mantle** (Responses API, `openai.gpt-5.6-sol`, Bearer key minted 8/17 — gate A DONE, credential-id `ACCAQLGEIFDRC5HTNASBQ`, key at `~/.config/bedrock-mantle/credentials.env`) · **4 = ChatGPT sub** (codex exec 0.147.0, `gpt-5.6-sol`). **Door 3 = OpenAI 1P** (api.openai.com Responses, `gpt-5.6-sol`, key saved 8/17 after one clipboard collision — gate B DONE, key at `~/.config/openai/credentials.env`). All raw receipts: `paper2-discovery-2026-08-17/` alongside this file. Costs: doors 1+2 ≈ ~$1 combined; door-4 $0 (free month), no rate limit hit at this volume.

## A. Door-1 (Converse) parameter acceptance — COMPLETE MATRIX

| Param | Shape tried | Verdict |
|---|---|---|
| `temperature` (0, 0.7) | inferenceConfig | REJECT — OpenAI-layer `unsupported_parameter` |
| `topP` | inferenceConfig | REJECT — `unsupported_parameter` (as `top_p`) |
| `seed` | additionalModelRequestFields flat | REJECT — `unknown_parameter` |
| `reasoning_effort` (flat) | AMRF flat | REJECT — `unknown_parameter` |
| `verbosity` (flat) | AMRF flat | REJECT — `unknown_parameter` |
| **`reasoning: {effort}`** (nested) | AMRF nested | **ACCEPT** |
| **`text: {verbosity}`** (nested) | AMRF nested | **ACCEPT** |

- ⇒ `supports_sampling: False` for Sol on the runtime door — same posture as the Claude-5 grid. No seed anywhere.
- **The passthrough speaks Responses-API shapes**: flat Chat-Completions names are *unknown*, nested Responses names work. (Instrument note: rejects surface as Bedrock `ValidationException` wrapping an OpenAI-style error JSON — the model layer, not Bedrock, is talking.)
- **Effort value set enumerated by the API itself** (unsupported_value error text): `none, low, medium, high, xhigh, max`. `minimal` is GONE in 5.6. Six-level ladder, mirrors Claude-5 naming.

## B. Effort is honored, and the DEFAULTS DIVERGE BY DOOR (paper-grade)

Prime-count probe ("primes strictly between 1000 and 1100" → 16, all arms correct):

| Arm | outputTokens | Note |
|---|---|---|
| Converse effort=none | 5 | answer only |
| Converse effort=low | 165 | reasoning billed inside outputTokens |
| Converse effort=high | 161 | |
| **Converse effort OMITTED** | **100** | **default ≠ none — nonzero, looks adaptive** |
| codex exec default | reasoning_output_tokens=0 | banner receipt `reasoning effort: none` |
| codex exec `-c model_reasoning_effort=high` | reasoning_output_tokens=91 | override works, answer 16 |

- **Same model, same weights: runtime door defaults to nonzero (adaptive-looking) reasoning; codex sub door defaults to `none`.** Ceiling-paper lineage, now receipted on both ends.
- Default-effort adaptivity: structured_json pilot ran at out=49/run (≈0 reasoning) while the prime probe default burned ~95 — the default scales to the task, so "default" is not a fixed arm. Prereg must pin effort explicitly per arm.
- **Converse returns NO reasoning content block** (content = text only; no reasoningContent like Claude-on-Bedrock) — reasoning on this door is *invisible but billed* in outputTokens. Usage carries `cacheReadInputTokens` (0 on these calls).

## C. n=20 repeat pilot (door defaults), frozen tasks from harness tasks.py

| Cell | distinct | modal share | headline |
|---|---|---|---|
| door1 extraction | 1 | 20/20 | byte-identical `PO-83614-QN`, 10 tok every run |
| door4 extraction | 1 | 20/20 | byte-identical, same 10 tok |
| door1 structured_json | 2 | 11/20 | fence 0/20 · all parse · ONE payload |
| door4 structured_json | 2 | 17/20 | fence 0/20 · all parse · ONE payload |
| door1 open_generation | 20 | 1/20 | max divergence; onset min char 51 / median 114; 546–713 out tok; 397–429 words |

- **THE finding — trailing-zero knife edge with door-dependent bias:** both doors emit exactly two byte-variants of the same JSON, differing ONLY at `unit_price_usd`: `349.50` vs `349.5` (first divergence char 102, outputTokens 49 both ways, both ≈0 reasoning). Mix by door: **Converse 11×`349.50`/9×`349.5` · codex 3×`349.50`/17×`349.5`** — modal output FLIPS across doors. Fisher exact two-sided p = 0.019 at pilot n. Same weights, same prompt, different door ⇒ different modal answer bytes. This is the Three-Doors thesis reproducing on OpenAI weights, on the production-realistic task.
- **Fence rate 0/20 on BOTH doors** — Paper #2's AWS-door fence headline does NOT reproduce on OpenAI weights (at defaults, this task). Fence pathology now reads as model-family-specific, not door-intrinsic — good contrast paragraph for the paper.
- Fragility ladder fully reproduces on OpenAI weights: 1 → 2 → 20 distinct outputs up the ladder (door1), exactly the Paper #2 ordering.

## D. Token accounting parity

- Output side agrees across doors on identical answers (49/49, 10/10) ⇒ same tokenizer view, comparable billing units.
- Input side: Converse in=150 (extraction task) vs codex in≈13.3K (9,984 cache-read) — **the Codex scaffold is ~13.3K tokens of standing input per exec call**; banner "tokens used" ≈ uncached-total (~3.2K trivial probe). Sharper than the 8/17 3,036 reading — that number was uncached-total, not scaffold size.
- codex usage fields: `input_tokens, cached_input_tokens, cache_write_input_tokens, output_tokens, reasoning_output_tokens`. Converse: `inputTokens, outputTokens, totalTokens, cacheReadInputTokens`.

## E. Context clamp (#33478) — REFUTED on 0.147.0; the real exec bound is a 1MB CHAR CAP

- `codex exec` rejects any prompt > **1,048,576 characters** at the protocol layer (JSON-RPC −32602 "exceeds the maximum length of 1048576 characters") — the model is never reached. Interactive sessions can accumulate more across turns; a single exec shot cannot. Own mini-finding: **the sub door cannot submit a 1M-token-scale prompt in one shot at all** (1M chars ≈ 150–280K tokens for English).
- 200K-char-budget control (tokenized to 127,795 actual input tokens — repetitive filler runs ~5.8 chars/tok): needle at char 0 retrieved exactly; no truncation at ~128K.
- **Dense probe (hex filler, 1,047,976 chars): 621,804 input tokens accepted in ONE exec call, needle retrieved exactly, reasoning 0, exit 0.** No ~258K clamp on this version — #33478 does not reproduce. Single-shot ceiling = the char cap (≈622K tokens at hex density, ~150–280K at English density).
- Prompt delivery receipt: `codex exec` accepts the prompt on stdin when no positional arg (verified `STDIN-OK`) — this is how anything above ARG_MAX travels.

## E2. Door-2 (mantle Responses) — full matrix, run 8/17 late evening

- **Invoke ✅** first try with the fresh key. Param posture IDENTICAL to runtime door: temperature/top_p `unsupported_parameter`, seed `unknown_parameter`, `minimal` rejected with the SAME enumerated set (none/low/medium/high/xhigh/max). **Flat `reasoning_effort` rejected on mantle too** — nested `reasoning:{effort}` only, on both Bedrock doors.
- **Superior instrumentation** — usage exposes `output_tokens_details.reasoning_tokens` + `input_tokens_details.{cached_tokens,cache_write_tokens}` (Converse hides the reasoning split).
- Effort behavior (prime probe): none→reason 0 · low→101 · high→147 · **default→101, byte-equal to the low arm** — "default=low?" is a prereg question (vs Converse default out=100 same prompt).
- n=20 pilot: extraction **20/20 byte-identical** (10 tok) · structured_json **2 variants, 13×`349.5` / 7×`349.50`** (49 tok, reasoning 0) · open_generation **20/20 distinct** (first-div min 51/median 58; 544–796 out tok; 399–435 words).
- **NEW: default-effort reasoning burn is run-variable — 55–282 reasoning tokens across 20 IDENTICAL open_generation requests.** The adaptive default is itself nondeterministic in spend; only visible through this door's usage detail. (structured_json burned 0 in all 20 — task-adaptive confirmed.)

### Three-door table on the trailing-zero knife edge (n=20/door, defaults)

| Door | `349.50` | `349.5` |
|---|---|---|
| 1 runtime Converse | 11 | 9 |
| 2 mantle Responses | 7 | 13 |
| 4 codex sub | 3 | 17 |

Same weights, same prompt: three doors, three biases — gradient from Converse toward codex. Ends differ at p=0.019 (Fisher, doors 1v4); adjacent pairs need the n=100 confirmatory. Extraction 20/20 identical on ALL THREE doors + reasoning=0 on every structured_json run ⇒ neither noise nor effort explains the flip.

## E3. Door-3 (OpenAI 1P Responses) — matrix + pilot, run 8/17 night

- **Invoke ✅ first try. Param posture IDENTICAL to both Bedrock doors**: temperature/top_p unsupported, seed unknown, `minimal` rejected with the SAME six-value enumeration (none/low/medium/high/xhigh/max). Instrument nuances: 1P's flat-`reasoning_effort` rejection helpfully says "moved to 'reasoning.effort', see docs" — **Bedrock's translation layer strips that guidance to a bare "Unknown parameter"**; 1P `temperature` error carries `"code":null` vs Bedrock's `"code":"unsupported_parameter"`.
- Door facts: `store:true` by DEFAULT (responses persisted server-side — mantle has no such field; prereg decides pin) · served model = undated alias `gpt-5.6-sol` (mantle: `openai.gpt-5.6-sol`) — **no dated snapshot on ANY door ⇒ silent point-version rolls invisible everywhere**, same as Claude-5 grid · `service_tier: default` both.
- n=20 pilot: extraction **20/20 byte-identical** · structured_json **11×`349.50`/9×`349.5`**, fence 0/20, reasoning 0 · open_generation **20/20 distinct** (first-div min 51/med 58, 401–441 words).
- **Reliability datum**: prime-default cell threw **4/15 transient 5xx** ("server had an error… retry") on the fresh key — Bedrock doors went 195/195 tonight. Single burst, not a rate claim; confirmatory runner needs retry logic on door 3.

### FOUR-door table — trailing-zero knife edge (n=20/door, defaults)

| Door | `349.50` | `349.5` |
|---|---|---|
| 3 OpenAI 1P | 11 | 9 |
| 1 runtime Converse | 11 | 9 |
| 2 mantle Responses | 7 | 13 |
| 4 codex sub | 3 | 17 |

**The two raw-API doors (1P + Converse) agree exactly; the two translated/wrapped doors drift progressively.** Ends differ at Fisher p=0.019 (n=20); middle pairs need n=100. Extraction 20/20 identical on ALL FOUR doors; reasoning 0 on every structured_json run everywhere.

### Default-effort CORRECTION (kills an early single-sample read)

Single prime probes had suggested 1P default ≈ high (373) vs Bedrock ≈ low (~100). **n=15/door says NO**: default-effort reasoning distributions on the prime prompt — door1 ~95/175/335 (total-out proxy) · door2 95/191/475 · door3 77/172/401 (n=11 after 5xx) — **statistically the same door to door**. The single samples were both tail draws of the same wide distribution. What IS real: (a) **codex `none` default remains the one structural outlier** (banner-receipted); (b) **adaptive-default reasoning burn varies ~4–5× across IDENTICAL requests on every API door** — the door bills you a different amount each run for the same question at defaults. Prereg: pin effort; treat "default" as its own labeled arm if kept.

## F. Also verified

- `global.openai.gpt-5.6-sol` invoke ✅ (in=11/out=5, 1,941ms) — both routing arms live for the us-vs-global sub-axis.

## G. What this sets up for prereg (v4 sketch inputs)

- Factors per door: door {runtime-us, runtime-global, mantle*, 1P*, codex-sub} × effort {none, low, medium, high, xhigh, max — PINNED, never default} × task ladder. (*gated on Joe: mantle key, 1P key.)
- `supports_sampling: False` runtime door; sampling acceptance on mantle/1P = first probe when keys arrive (1P GPT-5.x rejects temperature too; mantle unknown).
- structured_json trailing-zero cell is the confirmatory centerpiece: n=100/door, prereg the door-bias test (pilot p=0.019).
- Codex door: pin `-c model_reasoning_effort` explicitly; record banner receipt per call; 1MB char cap bounds any long-context cells on this door.

## Session log

- 8/17 evening (this session): §A–F run. Param matrix 16 calls · effort probes 8 · pilot 60+40 · clamp 4 · global 1. No failures, no rate limits.
- 8/17 late evening: gate A executed on Joe's go (mantle key minted + stored). §E2: door-2 matrix 13 calls + pilot 60. Door-3 key save attempt hit clipboard collision (file captured the command, not a key) — guarded re-save command issued to Joe.
- 8/17 night: gate B done (key saved inline, 164 chars). §E3: door-3 matrix 13 + pilot 60 (+4 5xx retries absorbed) · prime-default n=15×3 doors → default-effort correction. **DISCOVERY COMPLETE — all four doors characterized.** Total spend: doors 1+2+3 ≈ $2; door 4 $0. NEXT = prereg-v4 (~8/29).
