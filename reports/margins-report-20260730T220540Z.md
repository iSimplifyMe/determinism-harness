# Companion B report — logprob margins (exploratory)

Generated 2026-07-30T22:05:40.405638+00:00. Records in: 244 — warmups excluded: 4. Plan: FOLLOWUP-COMPANIONS.md (committed pre-data).

Observer caveat: logprobs request fields are NOT byte-neutral at generation length (freeze-gate probe): open_generation margins describe logprobs-perturbed trajectories, not the frozen runs'.

| box::cell | n | variants | min margin (min/med) | pos < 0.01 | fork idx | fork margins (modal/alt) |
|---|---|---|---|---|---|---|
| cuda::gpt-oss-20b\|open_generation\|greedy\|effort_low\|logprobs | 20 | 1 | 0.0111 / 0.0111 | 0.0000 | - | - |
| cuda::gpt-oss-20b\|structured_json\|greedy\|effort_low\|logprobs | 50 | 1 | 0.0183 / 0.0225 | 0.0000 | - | - |
| metal::gpt-oss-120b\|structured_json\|greedy\|effort_high\|logprobs | 50 | 2 | 0.0034 / 0.0034 | 0.0050 | 213 | 0.0190 / 0.0135 |
| metal::gpt-oss-120b\|structured_json\|greedy\|effort_low\|logprobs | 50 | 1 | 0.8462 / 0.8723 | 0.0000 | - | - |
| metal::gpt-oss-20b\|structured_json\|greedy\|effort_low\|logprobs | 50 | 1 | 0.0777 / 0.0777 | 0.0000 | - | - |
| metal::qwen3-vl-32b\|open_generation\|greedy\|think_off\|logprobs | 20 | 2 | 0.0014 / 0.0014 | 0.0019 | 199 | 0.0127 / 0.0046 |
