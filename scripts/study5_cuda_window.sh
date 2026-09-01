#!/bin/bash
# Study-5 CUDA window chain (attended; PREREGISTRATION-v5 sections 7-8).
# Usage: study5_cuda_window.sh pilot|full
#
# Owns the whole 4090 lifecycle, embodying every study-3 chain lesson:
#   - tunnel targets 127.0.0.1 explicitly (Windows resolves localhost
#     to ::1 and the tunnel dies)
#   - `ollama.exe serve` needs a held ssh session with a desktop -
#     detached Start-Process and CLI auto-start both die ("Unable to
#     init instance"); this script holds the session itself
#   - teardown is taskkill /IM ollama.exe /F over ssh (killing only the
#     ssh hold ORPHANS the server), then VERIFIED via tasklist AND
#     netstat before the tunnel comes down
#   - engine version + weights digest are captured by the runner
#     manifest (fail-fast if the box is not ready)
# Ordering gates identical to the API chain: no run on an unfrozen
# corpus; full refuses without the prereg-v5 tag.
set -u
MODE_ARG="${1:?usage: study5_cuda_window.sh pilot|full}"
case "$MODE_ARG" in
  pilot) MODE="study5-pilot-local"; EXPECT_CALLS=211 ;;
  full)  MODE="study5-full-local";  EXPECT_CALLS=1501 ;;
  *) echo "unknown mode: $MODE_ARG"; exit 2 ;;
esac
WINDOW="local"
HOST="${STUDY5_CUDA_HOST:-4090}"
LOCAL_PORT=11435
EXPECT_DIGEST="17052f91a42e"   # gpt-oss:20b, pinned equal across boxes
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 9
log() { echo "[study5-cuda-$MODE_ARG $(date -u +%H:%M:%SZ)] $*"; }

FROZEN=$(python3 -c "
import json; print(json.load(open('fixtures/study5/corpus.json'))['meta']['frozen'])")
if [ "$FROZEN" != "True" ]; then
  log "REFUSED: corpus meta.frozen is $FROZEN - freeze first."
  exit 3
fi
if [ "$MODE_ARG" = "full" ] && [ -z "$(git tag -l prereg-v5)" ]; then
  log "REFUSED: no prereg-v5 tag - confirmatory cannot precede the freeze."
  exit 3
fi
if lsof -iTCP:$LOCAL_PORT -sTCP:LISTEN >/dev/null 2>&1; then
  log "REFUSED: local port $LOCAL_PORT already in use."
  exit 3
fi

SERVE_PID=""
TUNNEL_PID=""
teardown() {
  log "teardown: taskkill ollama.exe on $HOST"
  ssh "$HOST" 'taskkill /IM ollama.exe /F' >/dev/null 2>&1
  sleep 2
  local left listeners
  left=$(ssh "$HOST" 'tasklist | findstr /I ollama' 2>/dev/null)
  listeners=$(ssh "$HOST" 'netstat -ano | findstr 11434 | findstr LISTENING' 2>/dev/null)
  if [ -n "$left" ] || [ -n "$listeners" ]; then
    log "WARNING: ollama still present on the box - verify by hand:"
    [ -n "$left" ] && echo "$left"
    [ -n "$listeners" ] && echo "$listeners"
  else
    log "box verified quiet (no process, no 11434 listener)"
  fi
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" >/dev/null 2>&1
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" >/dev/null 2>&1
  log "tunnel + serve holds released"
}
trap teardown EXIT

log "starting ollama serve on $HOST (held ssh session)"
ssh "$HOST" 'ollama.exe serve' >/dev/null 2>&1 &
SERVE_PID=$!
log "opening tunnel 127.0.0.1:$LOCAL_PORT -> $HOST:11434"
ssh -N -L "$LOCAL_PORT:127.0.0.1:11434" "$HOST" &
TUNNEL_PID=$!

READY=""
for _ in $(seq 1 30); do
  sleep 2
  if curl -s "http://127.0.0.1:$LOCAL_PORT/api/version" >/dev/null 2>&1; then
    READY=1; break
  fi
done
if [ -z "$READY" ]; then
  log "REFUSED: server never answered /api/version through the tunnel."
  exit 3
fi
DIGEST=$(curl -s "http://127.0.0.1:$LOCAL_PORT/api/tags" \
  | python3 -c "
import json,sys
for m in json.load(sys.stdin).get('models', []):
    if m.get('name') == 'gpt-oss:20b':
        print(m.get('digest', '')[:12])")
if [ "$DIGEST" != "$EXPECT_DIGEST" ]; then
  log "REFUSED: gpt-oss:20b digest '$DIGEST' != pinned $EXPECT_DIGEST."
  exit 3
fi
log "box ready (digest $DIGEST)"

DRY=$(python3 -m harness.runner --mode "$MODE" --window "$WINDOW" \
      --box cuda --local-url "http://127.0.0.1:$LOCAL_PORT" \
      --out runs --dry-run 2>&1)
if ! echo "$DRY" | grep -q "DRY RUN: $EXPECT_CALLS calls"; then
  log "REFUSED: dry-run count does not match registered $EXPECT_CALLS."
  echo "$DRY" | head -3
  exit 3
fi

log "starting $MODE ($EXPECT_CALLS calls, single-flight)"
python3 -m harness.runner --mode "$MODE" --window "$WINDOW" \
  --box cuda --local-url "http://127.0.0.1:$LOCAL_PORT" --out runs
RC=$?
log "runner exit $RC"

LATEST=$(ls -t runs/${WINDOW}-${MODE}-*.jsonl 2>/dev/null | head -1)
if [ -n "$LATEST" ] && [ "$RC" -lt 2 ]; then
  python3 -m analysis.analyze_study5 "$LATEST" --out reports
fi
log "done (teardown runs on exit). Commit runs/ + reports/ after review."
exit "$RC"
