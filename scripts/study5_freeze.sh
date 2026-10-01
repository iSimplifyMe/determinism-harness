#!/bin/bash
# Study-5 corpus FREEZE (PROTOCOL sections 6-7; PREREGISTRATION-v5 section 10,
# gate-2 exit). Runs ONLY on the owner's word ("freeze"). One dedicated,
# tagged commit flips `meta.frozen` in BOTH fixture corpora — primary
# (n=150) and natural (n=50) — and that commit must precede the first model
# call against any item (the chain scripts refuse until it exists).
#
# Usage: scripts/study5_freeze.sh [--dry-run] [--push] [--skip-agreement]
#   --dry-run         run every gate, report PASS/FAIL, change nothing
#   --push            after the tagged commit, push main + the tag
#   --skip-agreement  allow a null second-labeler agreement (owner call;
#                     the omission must be disclosed in the paper)
#
# Gates (all must PASS):
#   1. on main, clean tree, HEAD == origin/main (merge study5-fixtures first)
#   2. zero study-5 run records exist (freeze precedes the first call)
#   3. both corpora currently unfrozen; no freeze tag yet
#   4. natural arm re-derives from its snapshots (study5_natural check)
#   5. second-labeler agreement recorded (unless --skip-agreement)
#   6. full test suite green
# Then: flip both flags (exactly one occurrence each), re-validate, commit,
# annotated tag `study5-corpus-frozen`, optional push. Exit 3 on any FAIL.
set -u
DRY=""; PUSH=""; SKIP_AGREEMENT=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --push) PUSH=1 ;;
    --skip-agreement) SKIP_AGREEMENT=1 ;;
    *) echo "unknown argument: $arg"; exit 2 ;;
  esac
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 9
PRIMARY="fixtures/study5/corpus.json"
NATURAL="fixtures/study5/natural_corpus.json"
TAG="study5-corpus-frozen"
FAIL=0
log() { echo "[study5-freeze $(date -u +%H:%M:%SZ)] $*"; }
gate() {  # gate <name> <0|1 pass>
  if [ "$2" = "1" ]; then log "PASS  $1"; else log "FAIL  $1"; FAIL=1; fi
}

# --- 1. git state -----------------------------------------------------------
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] && gate "on main (got: $BRANCH)" 1 || gate "on main (got: $BRANCH)" 0
[ -z "$(git status --porcelain)" ] && gate "clean working tree" 1 || gate "clean working tree" 0
git fetch -q origin main 2>/dev/null
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main 2>/dev/null)" ]; then
  gate "HEAD == origin/main" 1
else
  gate "HEAD == origin/main (push or pull first)" 0
fi

# --- 2. no study-5 records --------------------------------------------------
RUNS="$(ls runs 2>/dev/null | grep -c study5 || true)"
[ "$RUNS" = "0" ] && gate "zero study-5 run records" 1 || gate "zero study-5 run records (found $RUNS)" 0

# --- 3. both unfrozen, no tag ----------------------------------------------
for corpus in "$PRIMARY" "$NATURAL"; do
  FROZEN=$(python3 -c "import json; print(json.load(open('$corpus'))['meta']['frozen'])")
  COUNT=$(grep -c '"frozen": false' "$corpus")
  if [ "$FROZEN" = "False" ] && [ "$COUNT" = "1" ]; then
    gate "$corpus unfrozen, single flag" 1
  else
    gate "$corpus unfrozen, single flag (frozen=$FROZEN, flags=$COUNT)" 0
  fi
done
[ -z "$(git tag -l "$TAG")" ] && gate "no $TAG tag yet" 1 || gate "no $TAG tag yet" 0

# --- 4. natural arm reproducible -------------------------------------------
if python3 -m harness.study5_natural check >/dev/null 2>&1; then
  gate "natural arm re-derives from snapshots" 1
else
  gate "natural arm re-derives from snapshots" 0
fi

# --- 5. second-labeler agreement -------------------------------------------
AGREEMENT=$(python3 -c "
import json; m = json.load(open('$NATURAL'))['meta']
print('yes' if m.get('labeling', {}).get('agreement') else 'no')")
if [ "$AGREEMENT" = "yes" ]; then
  gate "second-labeler agreement recorded" 1
elif [ -n "$SKIP_AGREEMENT" ]; then
  log "WARN  second-labeler agreement NULL - proceeding on --skip-agreement (disclose in the paper)"
else
  gate "second-labeler agreement recorded (run analysis.study5_agreement --write, or --skip-agreement)" 0
fi

# --- 6. suite ---------------------------------------------------------------
if python3 -m unittest discover -s tests -t . >/tmp/study5-freeze-suite.log 2>&1; then
  gate "full test suite green" 1
else
  gate "full test suite green (see /tmp/study5-freeze-suite.log)" 0
fi

if [ "$FAIL" = "1" ]; then
  log "REFUSED: gates failed - nothing changed."
  exit 3
fi
if [ -n "$DRY" ]; then
  log "DRY RUN: every gate passed - the real run would flip both flags, commit, tag $TAG${PUSH:+, push}."
  exit 0
fi

# --- flip -------------------------------------------------------------------
python3 - "$PRIMARY" "$NATURAL" <<'EOF'
import sys
for path in sys.argv[1:]:
    text = open(path, encoding="utf-8").read()
    assert text.count('"frozen": false') == 1, path
    open(path, "w", encoding="utf-8").write(text.replace('"frozen": false', '"frozen": true'))
    print(f"flipped {path}")
EOF
python3 - <<'EOF' || { log "REFUSED: post-flip validation failed - inspect the working tree (nothing committed)."; exit 3; }
from harness.study5_fixtures import (
    CORPUS_PATH, NATURAL_CORPUS_PATH, load_corpus, validate_corpus, validate_natural_corpus,
)
from harness.study5_natural import check_corpus
primary = load_corpus(CORPUS_PATH); natural = load_corpus(NATURAL_CORPUS_PATH)
assert primary["meta"]["frozen"] is True and natural["meta"]["frozen"] is True
assert validate_corpus(primary) == [], validate_corpus(primary)
assert validate_natural_corpus(natural, primary) == [], validate_natural_corpus(natural, primary)
assert check_corpus(natural) == [], check_corpus(natural)
print("post-flip validation clean")
EOF

# --- commit + tag -----------------------------------------------------------
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git add "$PRIMARY" "$NATURAL"
git commit -q -F - <<EOF
study5: FREEZE fixture corpora (primary n=150 + natural n=50) - gate 2 exit

PROTOCOL sections 6-7 / PREREGISTRATION-v5 section 10: meta.frozen flips
to true in both corpora in this one dedicated commit, ${STAMP}, before any
model call against any item. Labels, documents, templates, and provenance
are unchanged from the parent commit; this commit changes the two flags
only. The chain scripts accept runs from this commit onward; confirmatory
runs additionally wait for the prereg-v5 tag.
${FREEZE_TRAILER:+
$FREEZE_TRAILER}
EOF
git tag -a "$TAG" -m "Study-5 corpus freeze ${STAMP}: primary n=150 + natural n=50, flags only"
SHA="$(git rev-parse --short HEAD)"
log "FROZEN: commit $SHA, tag $TAG"
if [ -n "$PUSH" ]; then
  git push origin main && git push origin "$TAG" && log "pushed main + $TAG"
else
  log "not pushed (re-run with --push, or: git push origin main && git push origin $TAG)"
fi
exit 0
