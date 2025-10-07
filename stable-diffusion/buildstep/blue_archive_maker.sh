#!/usr/bin/env bash
set -euo pipefail

# 1) 환경변수 → 변수 세팅 (필요한 것만)
API_URL="${API_URL:-http://127.0.0.1:7860}"
OUT_ROOT="${OUT_ROOT:-$PWD/outputs}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STEPS="${STEPS:-28}"
SAMPLER="${SAMPLER:-Euler a}"
CFG_SCALE="${CFG_SCALE:-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_ITER="${N_ITER:-1}"
SEED="${SEED:--1}"
CHAR_LIST="${CHAR_LIST:-}"

[ -n "$CHAR_LIST" ] || { echo "CHAR_LIST 비어있음"; exit 1; }
mkdir -p "$OUT_ROOT"

# 2) 캐릭터 리스트 만큼 SD API 호출
IFS=',' read -r -a names <<< "$CHAR_LIST"

POS_TMPL='pixel_character_sprite, pxlchrctrsprt, sprite, sprite sheet, sprite art, pixel, (pixel art:1.5), retro game, retro, vibrant colors, pixelated, multiple views, concept art, (chibi:1.5), from side, looking away, from behind, back, 1girl, white background, {CHAR} (blue archive), has hair
<lora:pixel_character_sprite_illustrious:0.7>'
NEGATIVE='low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, copyright name, watermark, artist name, signature'

b64dec() { base64 -D 2>/dev/null || base64 -d 2>/dev/null || openssl base64 -d; }

for raw in "${names[@]}"; do
  char="$(echo "$raw" | sed 's/^ *//; s/ *$//')"
  [ -n "$char" ] || continue
  safe="$(echo "$char" | tr ' /' '__')"
  mkdir -p "${OUT_ROOT}/${safe}"

  prompt="${POS_TMPL//\{CHAR\}/$char}"

  req="$(jq -n \
    --arg prompt "$prompt" \
    --arg negative "$NEGATIVE" \
    --arg sampler "$SAMPLER" \
    --argjson steps "$STEPS" \
    --argjson width "$WIDTH" \
    --argjson height "$HEIGHT" \
    --argjson cfg "$CFG_SCALE" \
    --argjson batch "$BATCH_SIZE" \
    --argjson niter "$N_ITER" \
    --argjson seed "$SEED" \
    '{prompt:$prompt,negative_prompt:$negative,sampler_index:$sampler,steps:$steps,width:$width,height:$height,cfg_scale:$cfg,batch_size:$batch,n_iter:$niter,seed:$seed,save_images:false}'
  )"

  resp="$(curl -fsS -H 'Content-Type: application/json' \
          -X POST "${API_URL}/sdapi/v1/txt2img" \
          --data-binary "$req")"

  img_b64="$(echo "$resp" | jq -r '.images[0] // empty')"
  [ -n "$img_b64" ] || { echo "생성 실패: $char"; continue; }

  echo "$img_b64" | b64dec > "${OUT_ROOT}/${safe}/${safe}.png"
  echo "saved: ${OUT_ROOT}/${safe}/${safe}.png"
done