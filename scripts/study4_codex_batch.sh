#!/bin/bash
# Study-4 codex batch automation (PREREGISTRATION-v4 section 7;
# owner-approved 2026-08-17). Fired by launchd
# (com.isimplifyme.study4-codex-batch, StartInterval 18000 ~= the
# registered ~5h rate-window pacing). Each fire runs ONE batch through
# scripts/run_codex_batches.py (seed-stable resume, receipt-gated),
# commits and pushes the raw records (raw data is part of the artifact),
# and self-unloads the launchd job when the driver reports the schedule
# complete.
#
# Secrets: none in this file. SLACK_BOT_TOKEN / SLACK_CHANNEL arrive via
# the launchd plist's EnvironmentVariables (canary pattern); the codex
# door authenticates from the ChatGPT login state in ~/.codex.
set -u
REPO="$HOME/determinism-harness"
LABEL="com.isimplifyme.study4-codex-batch"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$REPO" || exit 9

git pull --ff-only >/dev/null 2>&1 \
  || echo "[study4 $STAMP] WARN: git pull failed; running on local main"

post_slack() {  # $1 icon  $2 message
  local token="${SLACK_BOT_TOKEN:-}"
  if [ -z "$token" ]; then
    token=$(/usr/libexec/PlistBuddy -c \
      "Print :EnvironmentVariables:SLACK_BOT_TOKEN" \
      "$HOME/Library/LaunchAgents/$LABEL.plist" 2>/dev/null || true)
  fi
  if [ -z "$token" ]; then
    echo "[study4 $STAMP] WARN: SLACK_BOT_TOKEN unavailable — logged only"
    return 0
  fi
  local payload
  payload=$(ICON="$1" MSG="$2" CH="${SLACK_CHANNEL:-C0AJBL89FGF}" python3 - <<'PYEOF'
import json, os
print(json.dumps({
    "channel": os.environ["CH"],
    "text": os.environ["ICON"] + " *Study-4 codex batches* " + os.environ["MSG"],
}))
PYEOF
)
  curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $token" \
    -H "Content-type: application/json; charset=utf-8" \
    --data "$payload" >/dev/null
}

PY="$REPO/.venv/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

OUT=$("$PY" -m scripts.run_codex_batches 2>&1)
RC=$?
echo "$OUT"

if echo "$OUT" | grep -q "SCHEDULE COMPLETE"; then
  echo "[study4 $STAMP] schedule complete — unloading $LABEL"
  post_slack ":checkered_flag:" \
    "SCHEDULE COMPLETE — all 800 codex calls attempted; launchd job unloading itself. Next: study4-q4q5 control window."
  # bootout kills this process group; detach so the exit is clean either way
  ( sleep 2; launchctl bootout "gui/$(id -u)/$LABEL" ) >/dev/null 2>&1 &
  exit 0
fi

if [ "$RC" -eq 3 ]; then
  post_slack ":rotating_light:" \
    "batch ABORTED — effort receipt mismatch (rc=3), no measured calls made; will retry next fire"
  exit 3
fi

PROGRESS=$(echo "$OUT" | grep -m1 "^progress:" || echo "progress: unknown")
NEW=$(git status --porcelain runs/ | grep -c "control-study4-codex" || true)
if [ "$NEW" -gt 0 ]; then
  git add runs/control-study4-codex-*
  git commit -q -m "study 4 confirmatory: codex batch (auto $STAMP) — $PROGRESS rc=$RC" \
    && git push -q origin main \
    && echo "[study4 $STAMP] records committed and pushed" \
    || echo "[study4 $STAMP] WARN: commit/push failed — records exist locally"
fi

case "$RC" in
  0) ICON=":white_check_mark:" ;;
  1) ICON=":warning:" ;;  # per-call failures = counted exclusions (never re-run)
  *) ICON=":rotating_light:" ;;
esac
post_slack "$ICON" "batch rc=$RC — $PROGRESS"
exit "$RC"
