#!/bin/bash
# Daily reproducibility canary (design doc 2026-07-30, owner-approved).
#
# Probes the three serving doors with the frozen task ladder slice and
# compares against committed baselines. Public-log policy: the log entry
# (canary/log/*.json) is committed every day; the raw run is committed
# only when status != GREEN (forensics when it matters, no daily bloat).
#
# Secrets: none in this file. The first-party key is pulled run-scoped
# from the macOS Keychain; ANTHROPIC_AWS_WORKSPACE_ID, SLACK_BOT_TOKEN,
# and SLACK_CHANNEL arrive via the launchd plist's EnvironmentVariables
# (com.isimplifyme.determinism-canary).
set -u
REPO="$HOME/determinism-harness"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$REPO" || exit 9

git pull --ff-only >/dev/null 2>&1 || echo "[canary $STAMP] WARN: git pull failed; running on local main"

post_slack() {  # $1 icon  $2 message
  local token="${SLACK_BOT_TOKEN:-}"
  if [ -z "$token" ]; then
    token=$(/usr/libexec/PlistBuddy -c \
      "Print :EnvironmentVariables:SLACK_BOT_TOKEN" \
      "$HOME/Library/LaunchAgents/com.isimplifyme.determinism-canary.plist" 2>/dev/null || true)
  fi
  if [ -z "$token" ]; then
    echo "[canary $STAMP] WARN: SLACK_BOT_TOKEN unavailable — logged only"
    return 0
  fi
  local payload
  payload=$(ICON="$1" MSG="$2" CH="${SLACK_CHANNEL:-C0AJBL89FGF}" python3 - <<'PYEOF'
import json, os
print(json.dumps({
    "channel": os.environ["CH"],
    "text": os.environ["ICON"] + " *Determinism canary* " + os.environ["MSG"],
}))
PYEOF
)
  local resp
  resp=$(curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $token" \
    -H "Content-type: application/json; charset=utf-8" \
    --data "$payload")
  case "$resp" in
    *'"ok":true'*) echo "[canary $STAMP] posted to Slack" ;;
    *) echo "[canary $STAMP] WARN: Slack post FAILED — $resp" ;;
  esac
}

# The Messages planes need the anthropic SDK — that lives in the repo's
# .venv (stdlib-only applies to analysis, not the API client layer). The
# 2026-07-30 inaugural rehearsal crashed on system python3 here.
PY="$REPO/.venv/bin/python3"
if [ ! -x "$PY" ]; then
  echo "[canary $STAMP] FATAL: $PY missing"
  post_slack ":rotating_light:" "FATAL — repo venv python missing; canary did not run"
  exit 3
fi

ANTHROPIC_API_KEY="$(security find-generic-password -a determinism -s anthropic-determinism-study -w 2>/dev/null)"
export ANTHROPIC_API_KEY
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "[canary $STAMP] FATAL: first-party key unavailable from Keychain"
  post_slack ":rotating_light:" "FATAL — first-party key unavailable; canary did not run"
  exit 3
fi
if [ -z "${ANTHROPIC_AWS_WORKSPACE_ID:-}" ]; then
  echo "[canary $STAMP] FATAL: ANTHROPIC_AWS_WORKSPACE_ID not set (plist env)"
  post_slack ":rotating_light:" "FATAL — workspace id unset; canary did not run"
  exit 3
fi

"$PY" -m harness.runner --mode canary --window canary --out runs
RUN_RC=$?
RUN=$(ls -t runs/canary-canary-*.jsonl 2>/dev/null | head -1)
# Fatal unless the runner completed (rc 0 = clean, 1 = per-call failures —
# still evaluable) AND left a non-empty record file. An rc-1 crash before
# any record is exactly what the rehearsal produced; require substance.
if [ "$RUN_RC" -ge 2 ] || [ -z "$RUN" ] || [ ! -s "$RUN" ]; then
  echo "[canary $STAMP] FATAL: runner rc=$RUN_RC run=${RUN:-none}"
  post_slack ":rotating_light:" "FATAL — runner rc=$RUN_RC produced no evaluable records"
  exit 2
fi

SUMMARY_FILE=$(mktemp)
"$PY" -m analysis.analyze_canary "$RUN" \
  --baselines canary/baselines.json --out canary/log | tee "$SUMMARY_FILE"
EVAL_RC=${PIPESTATUS[0]}
SUMMARY=$(grep -m1 "^CANARY " "$SUMMARY_FILE" || echo "CANARY UNKNOWN")
DETAIL=$(grep -E "^  (RED|YELLOW) " "$SUMMARY_FILE" | head -4)
rm -f "$SUMMARY_FILE"

case "$EVAL_RC" in
  0) ICON=":white_check_mark:" ;;
  1) ICON=":warning:" ;;
  *) ICON=":rotating_light:" ;;
esac

git add canary/log/
if [ "$EVAL_RC" -ne 0 ]; then
  BASE="${RUN%.jsonl}"
  git add "$RUN" "$BASE.manifest.json" "$BASE.done.json" 2>/dev/null
fi
git commit -q -m "canary: $(echo "$SUMMARY" | cut -c1-60) ($STAMP)" \
  && git push -q origin main \
  && echo "[canary $STAMP] log committed and pushed" \
  || echo "[canary $STAMP] WARN: commit/push failed — log exists locally"

MSG="$SUMMARY"
if [ -n "$DETAIL" ]; then
  MSG="$MSG
$DETAIL"
fi
post_slack "$ICON" "$MSG"
exit "$EVAL_RC"
