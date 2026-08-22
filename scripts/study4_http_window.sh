#!/bin/bash
# Study-4 HTTP-door confirmatory window (PREREGISTRATION-v4 section 7;
# spend owner-approved 2026-08-17; window timing = owner option A,
# 2026-08-21). One fire = ONE compressed `study4-full` run (2,400 calls /
# 24 cells) for the named window: low 07:00-10:00Z or peak 15:00-19:00Z.
#
# Fired by a ONE-SHOT launchd job (com.isimplifyme.study4-http-<window>,
# StartCalendarInterval pinned to a specific Month/Day so it cannot
# repeat; AbandonProcessGroup=true so the detached self-unload below
# actually runs — the codex loop's detached bootout was killed with the
# job's process group and never fired, 2026-08-22).
#
# Preflight is HARD (any failure => Slack + exit, nothing spent):
#   1. creds files present and non-empty (sourced at run time, never
#      stored in the plist)
#   2. interpreter imports boto3 + harness door modules
#   3. dry-run schedule = exactly 2,400 calls / 24 cells
#   4. start time inside the window's UTC band (a late/coalesced fire
#      after sleep must NOT run outside the registered band)
#   5. no prior non-dry-run record file for this window — a window is
#      never silently re-run; a partial window is a recorded deviation
#      (outline section 6 ledger), not a retry.
# Records, manifest and .done.json are committed + pushed (raw data is
# part of the artifact), same as the codex loop.
#
# Secrets: none in this file. SLACK_BOT_TOKEN / SLACK_CHANNEL arrive via
# the launchd plist's EnvironmentVariables (canary pattern).
set -u
WINDOW="${1:?usage: study4_http_window.sh low|peak}"
case "$WINDOW" in
  low)  BAND_START=7;  BAND_END=10 ;;   # [07,10)
  peak) BAND_START=15; BAND_END=19 ;;   # [15,19)
  *) echo "unknown window: $WINDOW"; exit 2 ;;
esac
REPO="$HOME/determinism-harness"
LABEL="com.isimplifyme.study4-http-$WINDOW"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EXPECT_CALLS=2400
EXPECT_CELLS=24
cd "$REPO" || exit 9

log() { echo "[study4-http-$WINDOW $STAMP] $*"; }

post_slack() {  # $1 icon  $2 message
  local token="${SLACK_BOT_TOKEN:-}"
  if [ -z "$token" ]; then
    # PlistBuddy prints "Does Not Exist" on STDOUT with rc=1 — only keep
    # the value on success, or a bogus bearer gets sent silently.
    token=$(/usr/libexec/PlistBuddy -c \
      "Print :EnvironmentVariables:SLACK_BOT_TOKEN" \
      "$HOME/Library/LaunchAgents/$LABEL.plist" 2>/dev/null) || token=""
  fi
  if [ -z "$token" ]; then
    log "WARN: SLACK_BOT_TOKEN unavailable — logged only"
    return 0
  fi
  local payload
  payload=$(ICON="$1" MSG="$2" CH="${SLACK_CHANNEL:-C0AJBL89FGF}" W="$WINDOW" python3 - <<'PYEOF'
import json, os
print(json.dumps({
    "channel": os.environ["CH"],
    "text": os.environ["ICON"] + " *Study-4 HTTP " + os.environ["W"] + " window* " + os.environ["MSG"],
}))
PYEOF
)
  curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $token" \
    -H "Content-type: application/json; charset=utf-8" \
    --data "$payload" >/dev/null
}

self_unload() {
  # one-shot job: retire the plist and boot the job out. Detached; the
  # plist sets AbandonProcessGroup=true so this survives our exit.
  local plist="$HOME/Library/LaunchAgents/$LABEL.plist"
  mkdir -p "$HOME/claude-headless/retired-plists"
  [ -f "$plist" ] && mv "$plist" "$HOME/claude-headless/retired-plists/$LABEL.plist.retired.$(date -u +%Y%m%dT%H%M%SZ)"
  ( sleep 3; launchctl bootout "gui/$(id -u)/$LABEL" ) >/dev/null 2>&1 &
}

abort() {  # $1 reason — nothing has been spent
  log "ABORT: $1"
  post_slack ":no_entry:" "ABORTED before any call — $1. Nothing spent; window NOT run. Manual decision required."
  self_unload
  exit 3
}

git pull --ff-only >/dev/null 2>&1 || log "WARN: git pull failed; running on local main"

# --- preflight 1: creds ---
for f in "$HOME/.config/openai/credentials.env" "$HOME/.config/bedrock-mantle/credentials.env"; do
  [ -s "$f" ] || abort "creds file missing/empty: $f"
  # shellcheck disable=SC1090
  . "$f"
done
export OPENAI_API_KEY BEDROCK_API_KEY
[ -n "${OPENAI_API_KEY:-}" ]  || abort "OPENAI_API_KEY empty after sourcing"
[ -n "${BEDROCK_API_KEY:-}" ] || abort "BEDROCK_API_KEY empty after sourcing"

# --- preflight 2: interpreter ---
PY="$REPO/.venv/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" -c "import boto3, harness.doors, harness.runner" >/dev/null 2>&1 \
  || abort "interpreter $PY cannot import boto3/harness"

# --- preflight 3: schedule shape (dry run, no HTTP) ---
DRY_DIR="$(mktemp -d)"
DRY=$("$PY" -m harness.runner --mode study4-full --window "$WINDOW" --dry-run --out "$DRY_DIR" 2>&1)
rm -rf "$DRY_DIR"
echo "$DRY" | grep -q "DRY RUN: $EXPECT_CALLS calls across $EXPECT_CELLS cells" \
  || abort "dry-run shape mismatch: $(echo "$DRY" | grep 'DRY RUN' || echo "$DRY" | tail -1)"

# --- preflight 4: inside the registered UTC band ---
HOUR=$((10#$(date -u +%H)))
if [ "$HOUR" -lt "$BAND_START" ] || [ "$HOUR" -ge "$BAND_END" ]; then
  abort "start hour ${HOUR}Z outside $WINDOW band [${BAND_START},${BAND_END})Z"
fi

# --- preflight 5: never re-run a window ---
PRIOR=$(ls runs/"$WINDOW"-study4-full-*.jsonl 2>/dev/null | grep -v dryrun || true)
[ -z "$PRIOR" ] || abort "window already has record file(s): $PRIOR"

post_slack ":rocket:" "STARTED $STAMP — $EXPECT_CALLS calls / $EXPECT_CELLS cells, one compressed run (prereg-v4 s7)."
log "preflight OK — starting live run"

OUT=$("$PY" -m harness.runner --mode study4-full --window "$WINDOW" 2>&1)
RC=$?
echo "$OUT"

RUN=$(echo "$OUT" | sed -n 's/^records -> runs\/\(.*\)\.jsonl$/\1/p' | head -1)
SUMMARY=""
if [ -n "$RUN" ] && [ -f "runs/$RUN.done.json" ]; then
  SUMMARY=$("$PY" - "runs/$RUN.done.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
keys = [k for k in ("done", "expected", "retries", "failures", "complete") if k in d]
print(" ".join(f"{k}={d[k]}" for k in keys))
PYEOF
)
fi
N=$( [ -n "$RUN" ] && [ -f "runs/$RUN.jsonl" ] && wc -l < "runs/$RUN.jsonl" | tr -d ' ' || echo 0 )

if [ -n "$RUN" ]; then
  git add "runs/$RUN".* 2>/dev/null
  git commit -q -m "study 4 confirmatory: $WINDOW window (auto $STAMP) — $N/$EXPECT_CALLS records rc=$RC $SUMMARY" \
    && git push -q origin HEAD:main \
    && log "records committed and pushed" \
    || log "WARN: commit/push failed — records on disk only"
fi

if [ "$RC" -eq 0 ] && [ "$N" -eq "$EXPECT_CALLS" ]; then
  post_slack ":white_check_mark:" "COMPLETE — $N/$EXPECT_CALLS records, $SUMMARY, rc=0, pushed. Job unloading itself."
else
  post_slack ":rotating_light:" "ENDED ABNORMALLY — rc=$RC, $N/$EXPECT_CALLS records on disk ($SUMMARY). DO NOT re-run; record as a deviation. Job unloading itself."
fi
self_unload
exit "$RC"
