#!/usr/bin/env bash
set -euo pipefail

# 1) 환경변수 → 변수 세팅
API_URL="${API_URL:-http://127.0.0.1:7860}"
OUT_ROOT="${OUT_ROOT:-$PWD/outputs}"          # 기본 출력 루트 경로
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-$(date +%F)}"  # 폴더 이름 (기본값: 오늘 날짜)
OUT_DIR="${OUT_ROOT%/}/${OUTPUT_DIR_NAME}"     # 최종 출력 경로

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

mkdir -p "$OUT_DIR"

# 유틸
need(){ command -v "$1" >/dev/null 2>&1 || { echo "필요 명령 없음: $1"; exit 1; }; }
need curl; need jq; need perl
b64dec(){ base64 -D 2>/dev/null || base64 -d 2>/dev/null || openssl base64 -d; }
sanitize_json(){ perl -pe 's/([\x00-\x1F])/sprintf("\\u%04X",ord($1))/ge'; }

# 2) 캐릭터 반복 호출
IFS=',' read -r -a names <<< "$CHAR_LIST"

POS_TMPL='pixel_character_sprite, pxlchrctrsprt, sprite, sprite sheet, sprite art, pixel, (pixel art:1.5), retro game, retro, vibrant colors, pixelated, multiple views, concept art, (chibi:1.5), from side, looking away, from behind, back, 1girl, white background, {CHAR} (blue archive), has hair
<lora:pixel_character_sprite_illustrious:0.7>'
NEGATIVE='low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, copyright name, watermark, artist name, signature'

for raw in "${names[@]}"; do
  char="$(echo "$raw" | sed 's/^ *//; s/ *$//')"
  [ -n "$char" ] || continue
  safe="$(echo "$char" | tr ' /' '__')"

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

  resp="$(curl -fsS -H 'Accept: application/json' -H 'Content-Type: application/json' \
          -X POST "${API_URL}/sdapi/v1/txt2img" --data-binary "$req")"

  img_b64="$(printf '%s' "$resp" | sanitize_json | jq -r '.images[0] // empty')"
  [ -n "$img_b64" ] || { echo "생성 실패: $char"; continue; }

  out_path="${OUT_DIR}/${safe}.png"
  echo "$img_b64" | b64dec > "$out_path"
  echo "saved: $out_path"
done

echo "✅ 완료: ${OUT_DIR}"