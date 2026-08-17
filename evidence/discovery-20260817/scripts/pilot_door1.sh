#!/bin/zsh
# Door-1 (Bedrock Converse) n=20 repeat pilot, 3 frozen tasks, door defaults.
set -u
SP="/private/tmp/claude-501/-Users-jelstner/4316d3c5-8bb4-4136-9f24-6a877a6bf1bf/scratchpad"
MODEL="us.openai.gpt-5.6-sol"
for task in extraction structured_json open_generation; do
  mkdir -p "$SP/pilot/door1/$task"
  MSGS=$(jq -n --rawfile t "$SP/task_$task.txt" '[{"role":"user","content":[{"text":$t}]}]')
  for i in $(seq 1 20); do
    f="$SP/pilot/door1/$task/$i.json"
    [[ -s "$f" ]] && continue
    if ! aws bedrock-runtime converse --profile default --region us-east-1 \
        --model-id "$MODEL" --messages "$MSGS" \
        --inference-config '{"maxTokens":16000}' --output json > "$f" 2>"$f.err"; then
      echo "FAIL $task/$i: $(head -c 150 "$f.err")"
      rm -f "$f"
      sleep 5
    fi
  done
  echo "done $task: $(ls "$SP/pilot/door1/$task"/*.json 2>/dev/null | wc -l | tr -d ' ')/20"
done
echo "DOOR1 PILOT COMPLETE"
