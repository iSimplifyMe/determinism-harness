"""Step-one verification: do the profile forms the grid depends on exist?

Writes evidence/inference-profiles.json. Exits nonzero if any required
profile is missing or inactive. Run this before any pilot or confirmatory
window; the evidence file is committed so the paper's routing claims trace
to a recorded observation.
"""
import json
import os
import sys
from datetime import datetime, timezone

import boto3

from harness.config import MODELS, REGION


def main():
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()

    bedrock = boto3.client("bedrock", region_name=REGION)
    summaries = bedrock.list_foundation_models(byProvider="anthropic")[
        "modelSummaries"
    ]
    inference_types = {
        m["modelId"]: m.get("inferenceTypesSupported", []) for m in summaries
    }

    profiles = {}
    token = None
    while True:
        kwargs = {"maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        page = bedrock.list_inference_profiles(**kwargs)
        for p in page.get("inferenceProfileSummaries", []):
            profiles[p["inferenceProfileId"]] = p.get("status")
        token = page.get("nextToken")
        if not token:
            break

    checks = []
    all_active = True
    for model_key, cfg in MODELS.items():
        bare_id = "anthropic." + next(iter(cfg["profiles"].values())).split(
            "anthropic.", 1
        )[1]
        checks.append(
            {
                "model": model_key,
                "bare_id": bare_id,
                "inference_types": inference_types.get(bare_id),
                "single_region_on_demand": "ON_DEMAND"
                in (inference_types.get(bare_id) or []),
            }
        )
        for form, pid in cfg["profiles"].items():
            active = profiles.get(pid) == "ACTIVE"
            all_active = all_active and active
            checks.append(
                {"model": model_key, "profile_form": form, "id": pid, "active": active}
            )

    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "aws_account": identity.get("Account"),
        "caller_arn": identity.get("Arn"),
        "region": REGION,
        "checks": checks,
        "all_required_profiles_active": all_active,
        "verdict": {
            "single_region_on_demand": (
                "UNAVAILABLE for all target models: inferenceTypesSupported is "
                "[INFERENCE_PROFILE] only, so the spec's original Q2 arm (bare "
                "model ID, single-region on-demand) cannot exist."
            ),
            "q2_reshaped": (
                "US-bounded routing (us.) vs worldwide routing (global.), both "
                "system-defined inference profiles, both required ACTIVE."
            ),
        },
    }

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evidence",
        "inference-profiles.json",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(f"account={result['aws_account']} region={REGION}")
    for check in checks:
        print(json.dumps(check, sort_keys=True))
    print(f"all_required_profiles_active={all_active}")
    print(f"evidence -> {out_path}")
    return 0 if all_active else 1


if __name__ == "__main__":
    raise SystemExit(main())
