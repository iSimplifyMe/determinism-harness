#!/bin/zsh
# Door-4 (ChatGPT sub via codex exec) n=20 repeat pilot, door defaults (effort none).
# structured_json first (fence-rate read), then extraction. Bail on rate limit.
set -u
SP="/private/tmp/claude-501/-Users-jelstner/4316d3c5-8bb4-4136-9f24-6a877a6bf1bf/scratchpad"
WORKDIR="$HOME/.cache/gpts"
for task in structured_json extraction; do
  mkdir -p "$SP/pilot/door4/$task"
  PROMPT="$(cat "$SP/task_$task.txt")"
  for i in $(seq 1 20); do
    f="$SP/pilot/door4/$task/$i.jsonl"
    [[ -s "$f" ]] && continue
    if ! codex exec --ephemeral -m gpt-5.6-sol -s read-only -C "$WORKDIR" \
        --skip-git-repo-check --json -- "$PROMPT" </dev/null > "$f" 2>"$f.err"; then
      echo "FAIL $task/$i (exit) — stderr tail:"
      tail -c 300 "$f.err"
      rm -f "$f"
      if grep -qiE 'rate.?limit|usage.?limit|429|too many' "$f.err"; then
        echo "RATE LIMITED — stopping sub pilot"
        exit 1
      fi
      sleep 10
    fi
  done
  echo "done $task: $(ls "$SP/pilot/door4/$task"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')/20"
done
echo "DOOR4 PILOT COMPLETE"
