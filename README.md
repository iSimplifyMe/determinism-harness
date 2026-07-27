# determinism-harness

Pre-registered measurement of AWS Bedrock inference reproducibility on models
that no longer expose sampling controls (`temperature`/`top_p`/`top_k` are
removed on the Claude 5 family — sending them returns a validation error,
verified live in `evidence/smoke.json`). The study asks four questions:

1. **Q1** — what is the exact-match reproduction rate for byte-identical
   requests, by model and task type?
2. **Q2** — does worldwide routing (`global.` inference profile) measurably
   change reproducibility versus US-bounded routing (`us.`)? There is no
   single-region on-demand path left to compare against: verified live,
   `inferenceTypesSupported = ["INFERENCE_PROFILE"]` for every study model
   (`evidence/inference-profiles.json`).
3. **Q3** — does adaptive thinking (the 5-family default) change final-answer
   reproducibility versus disabled thinking? The reasoning trace is never
   returned, but its size is measurable per call via
   `usage.output_tokens_details.thinking_tokens`.
4. **Q4** — does divergence correlate with time-of-day load?

Method commitments — equivalence margin, endpoints, exclusion rules, controls,
and what counts as a publishable null — are in [PREREGISTRATION.md](PREREGISTRATION.md)
(currently DRAFT; frozen by tag `prereg-v1` before any confirmatory data).
Execution steps are in [PROTOCOL.md](PROTOCOL.md). Measurement is deterministic
code end to end; no model is in the measurement loop.

## Layout

```
harness/
  config.py           frozen grid: models, routing profiles, arms, controls
  tasks.py            frozen task ladder (synthetic fixtures)
  request_builder.py  canonical request bytes; hashed bytes are sent bytes
  runner.py           scheduler + execution engine (ordering control, bounded
                      retries with attempt accounting, JSONL records)
  verify_profiles.py  live check that required routing profiles exist
  smoke.py            10-call instrument check incl. expected-rejection cases
analysis/
  stats.py            Wilson, exact binomial, two-proportion TOST (stdlib)
  metrics.py          modal share, divergence indices, banded Levenshtein
  analyze.py          gates -> per-cell metrics -> pre-registered comparisons
tests/                67 unit tests (stdlib unittest)
evidence/             committed live-verification artifacts
runs/                 manifests + per-call JSONL (committed after runs)
```

## Quickstart

```
python3 -m unittest discover -s tests -t .           # test suite
python3 -m harness.runner --mode pilot --window pilot --dry-run
python3 -m harness.verify_profiles                   # writes evidence/
python3 -m harness.smoke                             # ~10 live calls, writes evidence/
```

## Design at a glance

| | |
|---|---|
| Models | `us./global.anthropic.claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5-20251001-v1:0` |
| Grid | 40 cells/window: (2 models x 4 tasks x 2 profiles x 2 thinking) + (Haiku anchor x 4 tasks x 2 profiles) |
| Windows | low / mid / peak (UTC), compressed runs |
| Primary endpoint | per-cell modal share of byte-identical response text, Wilson 95% |
| Q2 test | TOST, delta = 1pp, alpha = 0.05, pooled per model |
| Positive control | Haiku 4.5 @ temperature 0.7 (the 5-family rejects sampling — cross-model control, disclosed) |
| Negative control | one request SHA-256 per cell; the hashed bytes are the sent bytes |

## Provenance

Built by iSimplifyMe. Companion artifacts: the published reference library at
isimplifyme.com/whitepapers and the `blind-panel` statistical-method repo this
harness follows (Wilson intervals, pre-registration, deterministic
verification). MIT license.
