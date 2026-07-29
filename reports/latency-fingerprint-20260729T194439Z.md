# Latency fingerprint per serving plane (study 2 addendum)

**EXPLORATORY / UNREGISTERED.** Post-hoc descriptive analysis of the study-2 confirmatory records' `latency_ms`; no confirmatory claims. Generated 2026-07-29T19:44:39.896135+00:00.

Records in: 13950 - eligible (ok, first attempt): 13845 - strata: 51.

## Covariates per plane

| plane | eligible | not ok | retried | inference_geo | service_tier |
|---|---|---|---|---|---|
| anthropic_api | 4644 | 0 | 6 | {'not_available': 900, 'global': 3744} | {'standard': 4644} |
| bedrock | 4649 | 0 | 1 | {'absent': 4649} | {'absent': 900, 'standard': 3749} |
| p_aws | 4552 | 85 | 13 | {'not_available': 805, 'global': 3747} | {'standard': 4552} |

## Decode rate - ms per generated token (binned-median WLS)

Fit over full-window, non-streamed records; output length is the lever arm (tasks range from a few tokens to open generation). `usage.output_tokens` includes thinking tokens on every plane, so the slope prices total decoded work. tok/s = 1000/slope.

| plane\|model\|window | ms/token | ~tok/s | intercept ms | n | bins | token range |
|---|---|---|---|---|---|---|
| anthropic_api|haiku-4-5|low | 12.3055 | 81.3 | 503.1 | 400 | 53 | 5-626 |
| anthropic_api|haiku-4-5|peak | 12.1933 | 82.0 | 474.2 | 400 | 56 | 5-625 |
| anthropic_api|opus-5|low | 14.3185 | 69.8 | 2439.5 | 795 | 189 | 6-1651 |
| anthropic_api|opus-5|peak | 14.3247 | 69.8 | 2390.8 | 799 | 179 | 6-1672 |
| anthropic_api|sonnet-5|low | 11.2647 | 88.8 | 1283.6 | 800 | 101 | 6-888 |
| anthropic_api|sonnet-5|peak | 11.0803 | 90.3 | 1291.4 | 800 | 100 | 6-880 |
| bedrock|haiku-4-5|low | 10.4913 | 95.3 | 657.6 | 400 | 53 | 5-625 |
| bedrock|haiku-4-5|peak | 10.5241 | 95.0 | 811.6 | 400 | 57 | 5-619 |
| bedrock|opus-5|low | 15.5771 | 64.2 | 1353.1 | 800 | 162 | 6-1508 |
| bedrock|opus-5|peak | 15.3526 | 65.1 | 1634.6 | 800 | 156 | 6-1520 |
| bedrock|sonnet-5|low | 11.8124 | 84.7 | 1358.6 | 799 | 105 | 6-893 |
| bedrock|sonnet-5|peak | 11.8327 | 84.5 | 1376.0 | 800 | 100 | 6-890 |
| p_aws|haiku-4-5|low | 12.1485 | 82.3 | 782.6 | 305 | 52 | 5-615 |
| p_aws|haiku-4-5|peak | 12.1506 | 82.3 | 757.3 | 400 | 58 | 5-610 |
| p_aws|opus-5|low | 14.7638 | 67.7 | 2618.2 | 798 | 160 | 6-1500 |
| p_aws|opus-5|peak | 14.499 | 69.0 | 2604.4 | 799 | 160 | 6-1659 |
| p_aws|sonnet-5|low | 11.4219 | 87.6 | 2231.0 | 800 | 103 | 6-907 |
| p_aws|sonnet-5|peak | 11.4257 | 87.5 | 2259.5 | 800 | 88 | 6-891 |

## Prefill rate - ms per input token (Q4 length ladder)

| plane\|model | ms/token | intercept ms | n | bins | token range |
|---|---|---|---|---|---|
| anthropic_api|opus-5 | -0.0008 | 2665.9 | 75 | 3 | 1472-55821 |
| anthropic_api|sonnet-5 | 0.0021 | 1414.1 | 75 | 3 | 1472-55821 |
| bedrock|opus-5 | 0.0154 | 1206.0 | 75 | 3 | 1474-55823 |
| bedrock|sonnet-5 | 0.0119 | 1710.3 | 75 | 3 | 1472-55821 |
| p_aws|opus-5 | 0.005 | 2914.8 | 75 | 3 | 1472-55821 |
| p_aws|sonnet-5 | 0.0067 | 2378.8 | 75 | 3 | 1472-55821 |

## Open-generation effective throughput (tokens/sec)

| plane\|model\|window | n | p25 | p50 | p75 |
|---|---|---|---|---|
| anthropic_api|haiku-4-5|low | 100 | 72.6 | 75.5 | 79.1 |
| anthropic_api|haiku-4-5|peak | 100 | 74.1 | 77.1 | 80.1 |
| anthropic_api|opus-5|low | 200 | 57.1 | 59.5 | 62.7 |
| anthropic_api|opus-5|peak | 200 | 57.9 | 60.1 | 62.8 |
| anthropic_api|sonnet-5|low | 200 | 75.9 | 78.5 | 80.5 |
| anthropic_api|sonnet-5|peak | 200 | 76.8 | 79.1 | 81.1 |
| bedrock|haiku-4-5|low | 100 | 82.9 | 85.0 | 90.2 |
| bedrock|haiku-4-5|peak | 100 | 80.4 | 83.0 | 86.4 |
| bedrock|opus-5|low | 200 | 57.3 | 59.1 | 60.8 |
| bedrock|opus-5|peak | 200 | 56.9 | 58.5 | 60.2 |
| bedrock|sonnet-5|low | 199 | 72.1 | 74.6 | 76.2 |
| bedrock|sonnet-5|peak | 200 | 71.8 | 74.1 | 76.3 |
| p_aws|haiku-4-5|low | 74 | 71.3 | 74.8 | 78.1 |
| p_aws|haiku-4-5|peak | 100 | 70.7 | 74.7 | 77.2 |
| p_aws|opus-5|low | 198 | 55.5 | 57.4 | 59.4 |
| p_aws|opus-5|peak | 200 | 56.2 | 57.9 | 60.0 |
| p_aws|sonnet-5|low | 200 | 67.8 | 71.4 | 74.4 |
| p_aws|sonnet-5|peak | 200 | 67.7 | 70.8 | 72.8 |

## Cross-plane distribution distances (two-sample KS)

`ks_centered` compares median-centered latencies - shape with the network/queue offset removed. `closest votes` counts strata where that pair is the most similar of the three (centered).

### full (40 strata with all three planes)

| pair | median KS raw | median KS centered | closest votes |
|---|---|---|---|
| anthropic_api|bedrock | 0.58 | 0.145 | 20 |
| anthropic_api|p_aws | 0.497 | 0.1643 | 14 |
| bedrock|p_aws | 0.83 | 0.2114 | 6 |

### positive_control (1 strata with all three planes)

| pair | median KS raw | median KS centered | closest votes |
|---|---|---|---|
| anthropic_api|bedrock | 0.59 | 0.12 | 0 |
| anthropic_api|p_aws | 0.19 | 0.16 | 0 |
| bedrock|p_aws | 0.68 | 0.11 | 1 |

### q4_lengths (6 strata with all three planes)

| pair | median KS raw | median KS centered | closest votes |
|---|---|---|---|
| anthropic_api|bedrock | 0.82 | 0.28 | 2 |
| anthropic_api|p_aws | 0.74 | 0.22 | 2 |
| bedrock|p_aws | 0.88 | 0.28 | 2 |

### streamed (4 strata with all three planes)

| pair | median KS raw | median KS centered | closest votes |
|---|---|---|---|
| anthropic_api|bedrock | 0.67 | 0.145 | 2 |
| anthropic_api|p_aws | 0.66 | 0.22 | 0 |
| bedrock|p_aws | 0.625 | 0.19 | 2 |

## Illustrative strata - median-centered latency shape

### peak::opus-5|structured_json|adaptive|request

| plane | n | p5 | p25 | p50 | p75 | p95 | IQR | CV |
|---|---|---|---|---|---|---|---|---|
| anthropic_api | 100 | 2418.2 | 2621.0 | 2963.5 | 3224.5 | 4329.2 | 603.5 | 0.4618 |
| bedrock | 100 | 2584.3 | 2670.8 | 2776.5 | 2991.5 | 4250.8 | 320.7 | 0.3805 |
| p_aws | 100 | 3587.1 | 3836.2 | 4008.5 | 4424.5 | 5284.3 | 588.3 | 0.1585 |

### peak::sonnet-5|structured_json|adaptive|request

| plane | n | p5 | p25 | p50 | p75 | p95 | IQR | CV |
|---|---|---|---|---|---|---|---|---|
| anthropic_api | 100 | 1537.9 | 1618.2 | 1729.5 | 1899.5 | 2205.5 | 281.3 | 0.2699 |
| bedrock | 100 | 1949.8 | 2034.0 | 2136.5 | 2247.5 | 3385.8 | 213.5 | 0.1702 |
| p_aws | 100 | 2589.8 | 2760.2 | 2936.5 | 3244.0 | 3922.0 | 483.8 | 0.2061 |

### peak::sonnet-5|open_generation|disabled|request

| plane | n | p5 | p25 | p50 | p75 | p95 | IQR | CV |
|---|---|---|---|---|---|---|---|---|
| anthropic_api | 100 | 9654.6 | 9952.5 | 10299.5 | 10706.5 | 11601.4 | 754.0 | 0.0614 |
| bedrock | 100 | 10359.5 | 10762.5 | 11120.0 | 11574.2 | 12191.1 | 811.7 | 0.0549 |
| p_aws | 100 | 10790.9 | 11186.2 | 11544.0 | 11960.2 | 12797.7 | 774.0 | 0.0633 |

## Caveats

- Exploratory and unregistered: written after the confirmatory results were known; no hypothesis was preregistered and no p-values are reported. Distances and slopes are descriptive.
- Raw latency mixes serving time with network path, TLS, and per-plane ingress overhead measured from one client in one city; raw-latency distances partly reflect geography, not serving hardware. Median-centered distances and per-token slopes are the informative readouts.
- P-AWS shed 85 calls (529) in the low window; its surviving low-window records may be load-biased (survivor bias). Bedrock and 1P had zero terminal failures.
- inference_geo covariate: both Messages planes (1P, P-AWS) report 'global' on 5-family records while Bedrock was pinned to the us. inference profile - regional routing is not held constant across planes by the platforms themselves.
- One client, one region-pinning configuration, two windows on one day per window type. Latency shapes may vary by day and region.
- A latency fingerprint cannot separate 'same hardware' from 'same software build on similar hardware'; agreement is consistent with a shared serving stack, not proof of one.
