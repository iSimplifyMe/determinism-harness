#!/bin/zsh
# Door-2 (Bedrock mantle, OpenAI Responses API) param matrix + effort probes.
set -u
source ~/.config/bedrock-mantle/credentials.env
SP="/private/tmp/claude-501/-Users-jelstner/4316d3c5-8bb4-4136-9f24-6a877a6bf1bf/scratchpad"
OUT="$SP/param_probes_door2"
mkdir -p "$OUT"

call() { # name, body
  local name="$1" body="$2"
  local resp
  resp=$(curl -sS --max-time 120 "$OPENAI_BASE_URL/responses" \
    -H "Authorization: Bearer $BEDROCK_API_KEY" -H "Content-Type: application/json" \
    -d "$body")
  printf '%s' "$resp" > "$OUT/$name.json"
  if printf '%s' "$resp" | jq -e '.error' >/dev/null 2>&1; then
    echo "REJECT | $name | $(printf '%s' "$resp" | jq -c '.error' | head -c 260)"
  else
    echo "ACCEPT | $name | $(printf '%s' "$resp" | jq -r '"in=\(.usage.input_tokens) out=\(.usage.output_tokens) reason=\(.usage.output_tokens_details.reasoning_tokens) ans=" + ([.output[]? | select(.type=="message") | .content[] | select(.type=="output_text") | .text] | join("") | .[0:25])')"
  fi
}

B='{"model":"openai.gpt-5.6-sol","input":"Reply with exactly: OK","max_output_tokens":512'
echo "=== matrix ==="
call base        "$B}"
call temp0       "$B,\"temperature\":0}"
call temp07      "$B,\"temperature\":0.7}"
call topp05      "$B,\"top_p\":0.5}"
call seed42      "$B,\"seed\":42}"
call eff_low     "$B,\"reasoning\":{\"effort\":\"low\"}}"
call eff_minimal "$B,\"reasoning\":{\"effort\":\"minimal\"}}"
call eff_flat    "$B,\"reasoning_effort\":\"low\"}"
call verb_low    "$B,\"text\":{\"verbosity\":\"low\"}}"

P='{"model":"openai.gpt-5.6-sol","input":"How many prime numbers are there strictly between 1000 and 1100? Answer with just the number.","max_output_tokens":16000'
echo "=== effort behavior (prime probe) ==="
call beh_default "$P}"
call beh_none    "$P,\"reasoning\":{\"effort\":\"none\"}}"
call beh_low     "$P,\"reasoning\":{\"effort\":\"low\"}}"
call beh_high    "$P,\"reasoning\":{\"effort\":\"high\"}}"
echo "MATRIX COMPLETE"
