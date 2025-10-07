#!/usr/bin/env bash
set -euo pipefail

# TeamCity env parameters -> shell env (without 'env.' prefix)
API_URL="${API_URL:-http://127.0.0.1:7860}"
OUT_ROOT="${OUT_ROOT:-/Users/kojeomstudio/stable-diffusion-webui/outputs}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STEPS="${STEPS:-28}"
SAMPLER="${SAMPLER:-Euler a}"
CFG_SCALE="${CFG_SCALE:-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_ITER="${N_ITER:-1}"
SEED="${SEED:--1}"
OVERWRITE="${OVERWRITE:-false}"

# 필수 파라미터(쉼표 구분)
if [[ -z "${CHAR_LIST:-}" ]]; then
  echo "ERROR: CHAR_LIST 파라미터가 비어있습니다. 예: Rikuhachima-Aru,Hoshino"
  exit 1
fi
IFS=',' read -r -a CHAR_LIST_ARR <<< "${CHAR_LIST}"

# 의존성 체크
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found"; exit 1; }; }
need curl
need jq

# base64 decode 명령 감지 (macOS 호환)
b64dec() {
  if base64 -D >/dev/null 2>&1 <<<"" ; then
    base64 -D
  elif base64 -d >/dev/null 2>&1 <<<"" ; then
    base64 -d
  else
    openssl base64 -d
  fi
}

# API 헬스 체크
if ! curl -fsS "${API_URL}/sdapi/v1/sd-models" >/dev/null; then
  echo "ERROR: Cannot reach SD WebUI API at ${API_URL}."
  echo "       에이전트 맥에서 ./webui.sh --api 로 실행되어 있는지 확인하세요."
  exit 1
fi

# 프롬프트 템플릿
read -r -d '' POSITIVE_TEMPLATE <<'EOF'
pixel_character_sprite, pxlchrctrsprt, sprite, sprite sheet, sprite art, pixel, (pixel art:1.5), retro game, retro, vibrant colors, pixelated, multiple views, concept art, (chibi:1.5), from side, looking away, from behind, back, 1girl, white background, {CHAR} (blue archive), has hair
<lora:pixel_character_sprite_illustrious:0.7>
EOF

NEGATIVE_PROMPT='low quality, worst quality, normal quality, text, signature, jpeg artifacts, bad anatomy, old, early, copyright name, watermark, artist name, signature'

save_image_b64() {
  local b64="$1" out_path="$2"
  echo "$b64" | b64dec > "$out_path"
}

unique_path_if_needed() {
  local path="$1"
  if [[ "$OVERWRITE" == "true" || ! -e "$path" ]]; then
    echo "$path"; return
  fi
  local base="${path%.*}" ext="${path##*.}" n=2
  while [[ -e "${base}_v${n}.${ext}" ]]; do n=$((n+1)); done
  echo "${base}_v${n}.${ext}"
}

# 숫자형 파라미터를 JSON 숫자로 넣기 위해 --argjson 사용(문자 들어오면 실패)
ensure_number() { [[ "$1" =~ ^-?[0-9]+$ ]] || { echo "ERROR: 숫자 아님 -> $2='$1'"; exit 1; }; }

ensure_number "$WIDTH"      "WIDTH"
ensure_number "$HEIGHT"     "HEIGHT"
ensure_number "$STEPS"      "STEPS"
ensure_number "$CFG_SCALE"  "CFG_SCALE"
ensure_number "$BATCH_SIZE" "BATCH_SIZE"
ensure_number "$N_ITER"     "N_ITER"
ensure_number "$SEED"       "SEED" || true  # -1 허용

# 생성 루프
for raw_name in "${CHAR_LIST_ARR[@]}"; do
  char="$(echo "$raw_name" | sed 's/^ *//; s/ *$//')"
  safe_char="$(echo "$char" | tr ' /' '__')"

  POSITIVE_PROMPT="${POSITIVE_TEMPLATE//\{CHAR\}/$char}"

  out_dir="${OUT_ROOT}/${safe_char}"
  mkdir -p "$out_dir"
  out_path="${out_dir}/${safe_char}.png"
  out_path="$(unique_path_if_needed "$out_path")"

  req_json="$(jq -n \
    --arg prompt "$POSITIVE_PROMPT" \
    --arg negative "$NEGATIVE_PROMPT" \
    --arg sampler "$SAMPLER" \
    --argjson steps "$STEPS" \
    --argjson width "$WIDTH" \
    --argjson height "$HEIGHT" \
    --argjson cfg "$CFG_SCALE" \
    --argjson batch "$BATCH_SIZE" \
    --argjson niter "$N_ITER" \
    --argjson seed "$SEED" \
    '{
      prompt: $prompt,
      negative_prompt: $negative,
      sampler_index: $sampler,
      steps: $steps,
      width: $width,
      height: $height,
      cfg_scale: $cfg,
      batch_size: $batch,
      n_iter: $niter,
      seed: $seed,
      save_images: false
    }'
  )"

  echo ">>> Generating '${char}' (${WIDTH}x${HEIGHT}, steps=${STEPS}, sampler=${SAMPLER}, cfg=${CFG_SCALE})"
  resp="$(curl -fsS -X POST "${API_URL}/sdapi/v1/txt2img" \
          -H "Content-Type: application/json" \
          --data-binary "$req_json" || true)"

  # 실패 감지
  if [[ -z "$resp" || "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "object" ]]; then
    echo "ERROR: API 응답 비정상. '${char}' 건 스킵"
    continue
  fi

  first_b64="$(echo "$resp" | jq -r '.images[0]' 2>/dev/null || true)"
  if [[ -z "$first_b64" || "$first_b64" == "null" ]]; then
    echo "ERROR: 이미지 생성 실패('${char}')"
    continue
  fi

  save_image_b64 "$first_b64" "$out_path"
  echo " ✔ saved: $out_path"
done

echo "✅ Done. Output root: ${OUT_ROOT}"