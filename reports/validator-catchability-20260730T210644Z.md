# Validator catchability (exploratory reanalysis)

Generated 2026-07-30T21:06:44.826606+00:00. Records analyzed: 33444 (of 33563 in; warmups excluded: 13; excluded: {'error': 106, 'truncated_or_other_stop': 0}). Zero new calls.

Validator model: structured_json: parse (strict|fenced) + exact schema; classification: label-set membership; extraction: format regex (format-only); open_generation: unvalidatable

## Pooled aggregates (validatable tasks)

| study::task::class | cells | n | byte modal | post-validator | reject | invisible |
|---|---|---|---|---|---|---|
| study1::classification::deterministic | 15 | 3493 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| study1::extraction::deterministic | 10 | 2995 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| study1::structured_json::deterministic | 10 | 2995 | 0.7649 | 0.9860 | 0.0000 | 0.0140 |
| study2::classification::deterministic | 15 | 2973 | 0.9966 | 0.9983 | 0.0017 | 0.0017 |
| study2::extraction::deterministic | 33 | 3428 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| study2::structured_json::deterministic | 21 | 3587 | 0.8157 | 0.9905 | 0.0000 | 0.0095 |
| study3::classification::deterministic | 9 | 900 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| study3::classification::sampled | 6 | 600 | 0.9283 | 0.9283 | 0.0000 | 0.0717 |
| study3::extraction::deterministic | 9 | 900 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| study3::extraction::sampled | 6 | 600 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| study3::structured_json::deterministic | 14 | 1400 | 0.9914 | 0.9993 | 0.0000 | 0.0007 |
| study3::structured_json::sampled | 6 | 600 | 0.5650 | 0.8033 | 0.0000 | 0.1967 |

## Cells with validator-relevant variance

| cell | class | n | byte modal | post-validator | reject | invisible | recovered |
|---|---|---|---|---|---|---|---|
| study1::haiku-4-5\|structured_json\|global\|none | deterministic | 300 | 0.8033 | 0.9467 | 0 | 0.0533 | 0.1433 |
| study1::haiku-4-5\|structured_json\|us\|none | deterministic | 300 | 0.7933 | 0.9133 | 0 | 0.0867 | 0.1200 |
| study1::opus-5\|structured_json\|global\|adaptive | deterministic | 300 | 0.5200 | 1.0000 | 0 | 0.0000 | 0.4800 |
| study1::opus-5\|structured_json\|global\|disabled | deterministic | 300 | 0.9300 | 1.0000 | 0 | 0.0000 | 0.0700 |
| study1::opus-5\|structured_json\|us\|adaptive | deterministic | 300 | 0.5467 | 1.0000 | 0 | 0.0000 | 0.4533 |
| study1::opus-5\|structured_json\|us\|disabled | deterministic | 300 | 0.9267 | 1.0000 | 0 | 0.0000 | 0.0733 |
| study1::sonnet-5\|structured_json\|global\|adaptive | deterministic | 300 | 0.5767 | 1.0000 | 0 | 0.0000 | 0.4233 |
| study1::sonnet-5\|structured_json\|global\|disabled | deterministic | 300 | 0.9733 | 1.0000 | 0 | 0.0000 | 0.0267 |
| study1::sonnet-5\|structured_json\|us\|adaptive | deterministic | 298 | 0.6040 | 1.0000 | 0 | 0.0000 | 0.3960 |
| study1::sonnet-5\|structured_json\|us\|disabled | deterministic | 297 | 0.9764 | 1.0000 | 0 | 0.0000 | 0.0236 |
| study2::haiku-4-5\|structured_json\|anthropic_api\|none | deterministic | 200 | 0.6050 | 0.9600 | 0 | 0.0400 | 0.3550 |
| study2::haiku-4-5\|structured_json\|bedrock\|none | deterministic | 200 | 0.8050 | 0.9600 | 0 | 0.0400 | 0.1550 |
| study2::haiku-4-5\|structured_json\|p_aws\|none | deterministic | 187 | 0.6203 | 0.9037 | 0 | 0.0963 | 0.2834 |
| study2::opus-5\|classification\|anthropic_api\|adaptive | deterministic | 200 | 0.9950 | 0.9950 | 0 | 0.0050 | 0.0000 |
| study2::opus-5\|classification\|anthropic_api\|disabled | deterministic | 200 | 0.9950 | 1.0000 | 1 | 0.0000 | 0.0050 |
| study2::opus-5\|classification\|bedrock\|adaptive | deterministic | 200 | 0.9950 | 0.9950 | 0 | 0.0050 | 0.0000 |
| study2::opus-5\|classification\|bedrock\|disabled | deterministic | 200 | 0.9950 | 1.0000 | 1 | 0.0000 | 0.0050 |
| study2::opus-5\|classification\|p_aws\|adaptive | deterministic | 200 | 0.9850 | 0.9850 | 0 | 0.0150 | 0.0000 |
| study2::opus-5\|classification\|p_aws\|disabled | deterministic | 200 | 0.9850 | 1.0000 | 3 | 0.0000 | 0.0150 |
| study2::opus-5\|structured_json\|anthropic_api\|adaptive | deterministic | 200 | 0.9850 | 1.0000 | 0 | 0.0000 | 0.0150 |
| study2::opus-5\|structured_json\|anthropic_api\|adaptive\|streamed | deterministic | 100 | 0.9800 | 1.0000 | 0 | 0.0000 | 0.0200 |
| study2::opus-5\|structured_json\|anthropic_api\|disabled | deterministic | 200 | 0.9800 | 1.0000 | 0 | 0.0000 | 0.0200 |
| study2::opus-5\|structured_json\|bedrock\|adaptive | deterministic | 200 | 0.8050 | 1.0000 | 0 | 0.0000 | 0.1950 |
| study2::opus-5\|structured_json\|bedrock\|adaptive\|streamed | deterministic | 100 | 0.6000 | 1.0000 | 0 | 0.0000 | 0.4000 |
| study2::opus-5\|structured_json\|bedrock\|disabled | deterministic | 200 | 0.9650 | 1.0000 | 0 | 0.0000 | 0.0350 |
| study2::opus-5\|structured_json\|p_aws\|adaptive | deterministic | 200 | 0.8200 | 1.0000 | 0 | 0.0000 | 0.1800 |
| study2::opus-5\|structured_json\|p_aws\|adaptive\|streamed | deterministic | 100 | 0.8100 | 1.0000 | 0 | 0.0000 | 0.1900 |
| study2::opus-5\|structured_json\|p_aws\|disabled | deterministic | 200 | 0.9800 | 1.0000 | 0 | 0.0000 | 0.0200 |
| study2::sonnet-5\|structured_json\|anthropic_api\|adaptive | deterministic | 200 | 0.8700 | 1.0000 | 0 | 0.0000 | 0.1300 |
| study2::sonnet-5\|structured_json\|anthropic_api\|adaptive\|streamed | deterministic | 100 | 0.8600 | 1.0000 | 0 | 0.0000 | 0.1400 |
| study2::sonnet-5\|structured_json\|bedrock\|adaptive | deterministic | 200 | 0.5600 | 1.0000 | 0 | 0.0000 | 0.4400 |
| study2::sonnet-5\|structured_json\|bedrock\|adaptive\|streamed | deterministic | 100 | 0.4900 | 1.0000 | 0 | 0.0000 | 0.5100 |
| study2::sonnet-5\|structured_json\|bedrock\|disabled | deterministic | 200 | 0.9750 | 1.0000 | 0 | 0.0000 | 0.0250 |
| study2::sonnet-5\|structured_json\|p_aws\|adaptive | deterministic | 200 | 0.5850 | 1.0000 | 0 | 0.0000 | 0.4150 |
| study2::sonnet-5\|structured_json\|p_aws\|adaptive\|streamed | deterministic | 100 | 0.5500 | 1.0000 | 0 | 0.0000 | 0.4500 |
| study2::sonnet-5\|structured_json\|p_aws\|disabled | deterministic | 200 | 0.9700 | 1.0000 | 0 | 0.0000 | 0.0300 |
| study3::cuda::gpt-oss-20b\|structured_json\|greedy\|effort_low | deterministic | 100 | 0.9900 | 0.9900 | 0 | 0.0100 | 0.0000 |
| study3::cuda::gpt-oss-20b\|structured_json\|temp07\|effort_low | sampled | 100 | 0.4900 | 0.5600 | 0 | 0.4400 | 0.0700 |
| study3::metal::gpt-oss-120b\|structured_json\|greedy\|effort_high | deterministic | 100 | 0.8900 | 1.0000 | 0 | 0.0000 | 0.1100 |
| study3::metal::gpt-oss-120b\|structured_json\|temp07\|effort_low | sampled | 100 | 0.6100 | 1.0000 | 0 | 0.0000 | 0.3900 |
| study3::metal::gpt-oss-20b\|structured_json\|temp07\|effort_low | sampled | 100 | 0.5800 | 0.6400 | 0 | 0.3600 | 0.0600 |
| study3::metal::qwen3-vl-32b\|structured_json\|temp07\|think_off | sampled | 100 | 0.9900 | 1.0000 | 0 | 0.0000 | 0.0100 |
| study3::metal::qwen3.5-122b\|structured_json\|temp07\|think_off | sampled | 100 | 0.3100 | 0.8800 | 0 | 0.1200 | 0.5700 |
| study3::metal::qwen3.6-35b\|classification\|temp07\|think_off | sampled | 100 | 0.5700 | 0.5700 | 0 | 0.4300 | 0.0000 |
| study3::metal::qwen3.6-35b\|structured_json\|temp07\|think_off | sampled | 100 | 0.4100 | 0.7400 | 0 | 0.2600 | 0.3300 |

## Unvalidatable coverage (open generation)

55 open-generation cells carry no deterministic validator; 48 are below byte ceiling — that instability is invisible to any output gate by construction.
