# Companion A report — reload-churn A/B (exploratory)

Generated 2026-07-30T22:05:25.394900+00:00. Cell: `gpt-oss-20b|open_generation|greedy|effort_low`. Records in: 402 — warmups excluded: 2. Plan: FOLLOWUP-COMPANIONS.md (committed pre-data).

## cuda

- cross-arm negative control (one sha both arms): True
- manipulation gate: pass=True (blocked_warm=True, churn_all_confirmed=True, churn_all_cold=True; blocked median load 0.37504125s, churn median 5.4083599499999995s)

| arm | n | modal share | distinct | matches confirmatory modal |
|---|---|---|---|---|
| blocked | 100 | 0.900 | 2 | False |
| churn | 100 | 1.000 | 1 | False |

Churn-minus-blocked modal-share diff: +0.1000 (SE 0.0300, CI95 [+0.0412, +0.1588]).

## metal

- cross-arm negative control (one sha both arms): True
- manipulation gate: pass=True (blocked_warm=True, churn_all_confirmed=True, churn_all_cold=True; blocked median load 0.2634847295s, churn median 1.6313302505s)

| arm | n | modal share | distinct | matches confirmatory modal |
|---|---|---|---|---|
| blocked | 100 | 0.900 | 2 | False |
| churn | 100 | 1.000 | 1 | False |

Churn-minus-blocked modal-share diff: +0.1000 (SE 0.0300, CI95 [+0.0412, +0.1588]).

## Cache-state decomposition (exploratory)

| box | position class | n | distinct | modal | prefill ms (med) | load s (med) |
|---|---|---|---|---|---|---|
| cuda | first_blocked_block | 10 | 1 | 1.000 | 37 | 0.38 |
| cuda | block_head | 9 | 1 | 1.000 | 18 | 0.38 |
| cuda | block_rest | 81 | 1 | 1.000 | 17 | 0.37 |
| cuda | churn | 100 | 1 | 1.000 | 199 | 5.41 |
| metal | first_blocked_block | 10 | 1 | 1.000 | 106 | 0.26 |
| metal | block_head | 9 | 1 | 1.000 | 30 | 0.26 |
| metal | block_rest | 81 | 1 | 1.000 | 29 | 0.26 |
| metal | churn | 100 | 1 | 1.000 | 118 | 1.63 |
