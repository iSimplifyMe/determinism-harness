#!/bin/bash
# Study-5 API run chain (attended; PREREGISTRATION-v5 sections 7-8).
# Usage: study5_api_run.sh pilot|full|natural
#
# This script IS the ordering enforcement:
#   - any run refuses while its fixture corpus is unfrozen
#     (corpus-freeze commit precedes the first model call against items)
#   - a FULL or NATURAL (confirmatory) run additionally refuses until the
#     `prereg-v5` tag exists (freeze precedes the first confirmatory call)
#   - natural = the held-out natural arm (PROTOCOL section 7): 50 items
#     x 5 templates x {haiku_1p, sonnet_1p} + 50 x 5 haiku resample = 750
# Preflight is hard: creds resolve, AWS identity answers, and the dry-run
# schedule matches the registered call count, else nothing is spent.
#
# Secrets: 1P key pulled run-scoped from the macOS Keychain (canary
# pattern) - never exported to the environment of anything but the
# runner, never printed, never in this file.
set -u
MODE_ARG="${1:?usage: study5_api_run.sh pilot|full|natural}"
CORPUS="fixtures/study5/corpus.json"
case "$MODE_ARG" in
  pilot)   MODE="study5-pilot-api";   EXPECT_CALLS=420;  WINDOW="pilot" ;;
  full)    MODE="study5-full-api";    EXPECT_CALLS=3000; WINDOW="peak" ;;
  natural) MODE="study5-natural-api"; EXPECT_CALLS=750;  WINDOW="peak"
           CORPUS="fixtures/study5/natural_corpus.json" ;;
  *) echo "unknown mode: $MODE_ARG"; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 9
log() { echo "[study5-api-$MODE_ARG $(date -u +%H:%M:%SZ)] $*"; }

# --- Gate 1: corpus frozen -------------------------------------------------
FROZEN=$(python3 -c "
import json; print(json.load(open('$CORPUS'))['meta']['frozen'])")
if [ "$FROZEN" != "True" ]; then
  log "REFUSED: $CORPUS meta.frozen is $FROZEN - freeze the corpus (its own"
  log "tagged commit) before any model call against corpus items."
  exit 3
fi

# --- Gate 2 (confirmatory: full + natural): prereg tag exists --------------
if [ "$MODE_ARG" = "full" ] || [ "$MODE_ARG" = "natural" ]; then
  if [ -z "$(git tag -l prereg-v5)" ]; then
    log "REFUSED: no prereg-v5 tag - the confirmatory run cannot precede"
    log "the frozen preregistration."
    exit 3
  fi
fi

# --- Preflight: creds + identity + dry-run count ---------------------------
KEY="$(security find-generic-password -a determinism -s anthropic-determinism-study -w 2>/dev/null)"
if [ -z "$KEY" ]; then
  log "REFUSED: 1P key not in Keychain (account determinism, service"
  log "anthropic-determinism-study)."
  exit 3
fi
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  log "REFUSED: AWS identity unavailable (Bedrock plane needs it)."
  exit 3
fi
DRY=$(python3 -m harness.runner --mode "$MODE" --window "$WINDOW" \
      --out runs --dry-run 2>&1)
echo "$DRY" | sed -n '1,2p'
if ! echo "$DRY" | grep -q "DRY RUN: $EXPECT_CALLS calls"; then
  log "REFUSED: dry-run call count does not match registered $EXPECT_CALLS."
  exit 3
fi

# --- Run -------------------------------------------------------------------
log "starting $MODE ($EXPECT_CALLS calls)"
ANTHROPIC_API_KEY="$KEY" python3 -m harness.runner \
  --mode "$MODE" --window "$WINDOW" --out runs
RC=$?
log "runner exit $RC (0=clean, 1=had failures, 2=incomplete, 3=creds)"
[ "$RC" -ge 2 ] && exit "$RC"

# --- Analyze ---------------------------------------------------------------
LATEST=$(ls -t runs/${WINDOW}-${MODE}-*.jsonl 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  log "analyzing $LATEST"
  python3 -m analysis.analyze_study5 "$LATEST" --out reports
fi
log "done. Next: review the report, then commit runs/ + reports/ (raw"
log "records are part of the public artifact - studies 1-4 pattern)."
exit "$RC"
