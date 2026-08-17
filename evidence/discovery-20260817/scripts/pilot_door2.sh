#!/bin/zsh
# Door-2 (mantle Responses) n=20 repeat pilot, 3 frozen tasks, door defaults.
set -u
source ~/.config/bedrock-mantle/credentials.env
SP="/private/tmp/claude-501/-Users-jelstner/4316d3c5-8bb4-4136-9f24-6a877a6bf1bf/scratchpad"
for task in extraction structured_json open_generation; do
  mkdir -p "$SP/pilot/door2/$task"
  BODY=$(jq -n --rawfile t "$SP/task_$task.txt" '{"model":"openai.gpt-5.6-sol","input":$t,"max_output_tokens":16000}')
  for i in $(seq 1 20); do
    f="$SP/pilot/door2/$task/$i.json"
    [[ -s "$f" ]] && continue
    RESP=$(curl -sS --max-time 180 "$OPENAI_BASE_URL/responses" \
      -H "Authorization: Bearer $BEDROCK_API_KEY" -H "Content-Type: application/json" \
      -d "$BODY")
    if printf '%s' "$RESP" | jq -e '.usage' >/dev/null 2>&1; then
      printf '%s' "$RESP" > "$f"
    else
      echo "FAIL $task/$i: $(printf '%s' "$RESP" | head -c 200)"
      sleep 5
    fi
  done
  echo "done $task: $(ls "$SP/pilot/door2/$task"/*.json 2>/dev/null | wc -l | tr -d ' ')/20"
done
echo "DOOR2 PILOT COMPLETE"
