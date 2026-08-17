#!/bin/zsh
# Door-4 context-clamp probe (openai/codex#33478: exec ~258K vs 1.05M interactive).
# Needle at char 0, retrieval instruction at end. 200K control, 300K test.
set -u
SP="/private/tmp/claude-501/-Users-jelstner/4316d3c5-8bb4-4136-9f24-6a877a6bf1bf/scratchpad"
WORKDIR="$HOME/.cache/gpts"
COMMON=(--ephemeral -m gpt-5.6-sol -s read-only -C "$WORKDIR" --skip-git-repo-check --json)

echo "=== stdin delivery check (tiny) ==="
print -r -- "Reply with exactly: STDIN-OK" | codex exec "${COMMON[@]}" > "$SP/clamp_stdin_check.jsonl" 2>"$SP/clamp_stdin_check.err"
echo "exit=$? answer=$(jq -r 'select(.type=="item.completed") | .item | select(.type=="agent_message") | .text' "$SP/clamp_stdin_check.jsonl" 2>/dev/null | tail -1)"

for label in 200k 300k; do
  echo "=== clamp $label ==="
  codex exec "${COMMON[@]}" < "$SP/clamp_$label.txt" > "$SP/clamp_$label.jsonl" 2>"$SP/clamp_$label.err"
  rc=$?
  ans=$(jq -r 'select(.type=="item.completed") | .item | select(.type=="agent_message") | .text' "$SP/clamp_$label.jsonl" 2>/dev/null | tail -1)
  usage=$(grep -o '"usage":{[^}]*}' "$SP/clamp_$label.jsonl" | tail -1)
  echo "exit=$rc"
  echo "answer: ${ans:0:120}"
  echo "usage: $usage"
  errs=$(grep -io 'context[^"]*\|exceed[^"]*\|too long[^"]*\|maximum[^"]*' "$SP/clamp_$label.jsonl" "$SP/clamp_$label.err" 2>/dev/null | head -5)
  [[ -n "$errs" ]] && { echo "limit-ish strings:"; echo "$errs"; }
done
echo "CLAMP PROBE COMPLETE"
