#!/bin/zsh
# Door-1 (Bedrock Converse) param acceptance matrix for openai.gpt-5.6-sol
# Each probe: tiny prompt, capture accept/reject + usage. ~cents total.
MODEL="us.openai.gpt-5.6-sol"
OUT="/private/tmp/claude-501/-Users-jelstner/4316d3c5-8bb4-4136-9f24-6a877a6bf1bf/scratchpad/param_probes"
mkdir -p "$OUT"
MSGS='[{"role":"user","content":[{"text":"Reply with exactly: OK"}]}]'

probe() {
  local name="$1" icfg="$2" amrf="$3"
  local args=(--profile default --region us-east-1 --model-id "$MODEL" --messages "$MSGS" --output json)
  [[ -n "$icfg" ]] && args+=(--inference-config "$icfg")
  [[ -n "$amrf" ]] && args+=(--additional-model-request-fields "$amrf")
  if RESP=$(aws bedrock-runtime converse "${args[@]}" 2>"$OUT/$name.err"); then
    printf '%s\n' "$RESP" > "$OUT/$name.json"
    local text=$(printf '%s' "$RESP" | jq -r '[.output.message.content[] | select(.text) | .text] | join("")' | head -c 40)
    local usage=$(printf '%s' "$RESP" | jq -r '"in=\(.usage.inputTokens) out=\(.usage.outputTokens) stop=\(.stopReason)"')
    echo "ACCEPT | $name | $usage | text=${text}"
  else
    echo "REJECT | $name | $(head -c 300 "$OUT/$name.err" | tr '\n' ' ')"
  fi
}

echo "=== baseline ==="
probe base '{"maxTokens":512}' ''
echo "=== sampling in inferenceConfig ==="
probe temp0        '{"maxTokens":512,"temperature":0}' ''
probe temp07       '{"maxTokens":512,"temperature":0.7}' ''
probe topp05       '{"maxTokens":512,"topP":0.5}' ''
probe temp_topp    '{"maxTokens":512,"temperature":0,"topP":1.0}' ''
echo "=== additionalModelRequestFields ==="
probe seed42       '{"maxTokens":512}' '{"seed":42}'
probe eff_low      '{"maxTokens":512}' '{"reasoning_effort":"low"}'
probe eff_minimal  '{"maxTokens":512}' '{"reasoning_effort":"minimal"}'
probe eff_none     '{"maxTokens":512}' '{"reasoning_effort":"none"}'
probe eff_high     '{"maxTokens":512}' '{"reasoning_effort":"high"}'
probe verb_low     '{"maxTokens":512}' '{"verbosity":"low"}'
echo "=== combos ==="
probe temp_efflow  '{"maxTokens":512,"temperature":0}' '{"reasoning_effort":"low"}'
probe seed_efflow  '{"maxTokens":512}' '{"seed":42,"reasoning_effort":"low"}'
