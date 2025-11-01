#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
import requests
import argparse
import base64

# --- TeamCity service message helpers ---
def _tc_escape(s: str) -> str:
    if s is None:
        return ""
    return (s.replace('|', '||')
             .replace("'", "|'")
             .replace('\n', '|n')
             .replace('\r', '|r')
             .replace(']', '|]')
             .replace('[', '|['))

def tc_message(text: str, status: str = "NORMAL"):
    print(f"##teamcity[message text='{_tc_escape(str(text))}' status='{_tc_escape(status)}']", flush=True)

def tc_progress(text: str):
    print(f"##teamcity[progressMessage '{_tc_escape(str(text))}']", flush=True)

def tc_problem(description: str):
    print(f"##teamcity[buildProblem description='{_tc_escape(str(description))}']", flush=True)

def try_unload_checkpoint(base_url: str, timeout: int = 60):
    # 1) 현재 메모리 상태 로깅 (선택)
    try:
        mem = requests.get(f"{base_url}/sdapi/v1/memory", timeout=timeout).json()
        tc_message(f"Memory(before)={mem}")
    except Exception:
        pass

    # 2) 체크포인트 언로드
    resp = requests.post(f"{base_url}/sdapi/v1/unload-checkpoint", timeout=timeout)
    resp.raise_for_status()
    tc_message("Checkpoint unloaded")

    # 3) 언로드 후 메모리 재확인 (선택)
    try:
        mem2 = requests.get(f"{base_url}/sdapi/v1/memory", timeout=timeout).json()
        tc_message(f"Memory(after)={mem2}")
    except Exception:
        pass

def getenv(name, default=None, cast=None):
    val = os.environ.get(name, default)
    if cast and val is not None:
        try:
            return cast(val)
        except Exception:
            return default
    return val

def _decode_base64_image(b64_str: str) -> bytes:
    # 일부 배포본은 "data:image/png;base64,..." 접두사가 붙을 수 있으므로 분리 처리
    if "," in b64_str and b64_str.strip().lower().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)

def _parse_csv_tags(s: str):
    """
    콤마(,)로 구분된 태그 문자열을 리스트로 변환.
    공백/빈 항목 제거.
    예) "<lora:Foo:1>, <lora:Bar:0.8>" -> ["<lora:Foo:1>", "<lora:Bar:0.8>"]
    """
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

def main():
    parser = argparse.ArgumentParser(description="실행 인자")
    parser.add_argument("--config", type=str, default="config.json", help="프롬프트 설정 파일 경로 (기본: config.json)")

    args = parser.parse_args()
    args_config_path = Path(args.config)

    # --- 1) 경로/환경 ---
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / args_config_path

    host = getenv("SD_HOST", "127.0.0.1")
    port = getenv("SD_PORT", "7860")
    base_url = f"http://{host}:{port}"

    outdir_root = Path(getenv("OUTDIR", "out"))
    outdir_root.mkdir(parents=True, exist_ok=True)

    # --- 2) 공통 인자 (빌드스텝 환경변수) ---
    model = getenv("MODEL", "")                         # 선택
    sampler = getenv("SAMPLER", "Euler a")
    steps = getenv("STEPS", None, cast=int)             # ✅ 필수 권장(전역만 사용)
    if steps is None or steps <= 0:
        tc_problem("env.STEPS must be a positive integer")
        raise SystemExit(1)

    cfg_scale = getenv("CFGSCALE", "7.0", cast=float)   # ✅ 전역 CFGSCALE
    width = getenv("WIDTH", "512", cast=int)
    height = getenv("HEIGHT", "512", cast=int)
    negative_prompt = getenv("NEGATIVE_PROMPT", "")
    seed = getenv("SEED", "-1", cast=int)
    batch_size = getenv("BATCH_SIZE", "1", cast=int)

    hr_second_pass = getenv("HR_SECOND_PASS", "false").lower() == "true"
    hr_upscale = getenv("HR_UPSCALE", "1.5", cast=float)
    hr_upscaler = getenv("HR_UPSCALER", "Latent (nearest-exact)")
    denoising_strength = getenv("DENOISING_STRENGTH", "0.4", cast=float)

    timeout = getenv("TIMEOUT_SEC", "180", cast=int)

    # --- 2-1) LoRA 태그(환경변수) 처리 ---
    # 예: LORA_TAGS="<lora:Foo:1>,<lora:Bar:0.8>"
    lora_tags_env = getenv("LORA_TAGS", "") or ""
    lora_tags = _parse_csv_tags(lora_tags_env)
    if lora_tags:
        tc_message(f"LoRA tags from env: {', '.join(lora_tags)}")

    # --- 3) config 로드 (딕셔너리만 허용) ---
    if not config_path.exists():
        tc_problem("config.json not found")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    prompts = config.get("prompts", {})
    if not isinstance(prompts, dict) or not prompts:
        tc_problem("prompts must be a non-empty object (key=name, value=prompt string)")
        raise SystemExit(1)

    total_prompts_num = len(prompts)
    cur_prompt_idx = 0

    # --- 4) 프롬프트 루프 (모든 프롬프트에 동일 STEPS/CFGSCALE 적용) ---
    for key, prompt in prompts.items():
        if not isinstance(prompt, str) or not prompt.strip():
            tc_message(f"Skip empty prompt for key {key}", status="WARNING")
            continue

        # 4-1) 프롬프트 뒤에 LORA 태그 append
        # - 프롬프트 끝의 불필요한 콤마/공백 제거 후 ", " + 태그들
        final_prompt = prompt.strip()
        if lora_tags:
            if final_prompt.endswith(","):
                final_prompt = final_prompt.rstrip(", \t")
            final_prompt = (final_prompt + ", " if final_prompt else "") + ", ".join(lora_tags)

        outdir_string = outdir_root
        outdir_string.mkdir(parents=True, exist_ok=True)

        # 서버 저장은 끄고, 전송만 받는다.
        # 파일명 패턴은 되돌릴 때를 대비해 유지하지만, 실제 저장은 클라이언트에서 수행.
        filename_pattern = f"[none]{key}"

        payload = {
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,                  # ✅ 전역 STEPS
            "cfg_scale": cfg_scale,          # ✅ 전역 CFGSCALE
            "width": width,
            "height": height,
            "seed": seed,
            "sampler_name": sampler,
            "batch_size": batch_size,

            "save_images": False,            # ✅ 서버 저장 X
            "send_images": True,             # ✅ 클라이언트로 base64 전송

            "enable_hr": hr_second_pass,
            "hr_scale": hr_upscale if hr_second_pass else 1.0,
            "hr_upscaler": hr_upscaler if hr_second_pass else None,
            "denoising_strength": denoising_strength if hr_second_pass else None,

            "override_settings": {
                **({"sd_model_checkpoint": model} if model else {}),
                "outdir_txt2img_samples": str(outdir_string),
                "samples_filename_pattern": filename_pattern,
                "save_to_dirs": False,
                "save_images_add_number": False
            },
            "override_settings_restore_afterwards": True
        }

        cur_prompt_idx += 1

        try:
            tc_message(f"Generating key={key} -> {outdir_string}")
            tc_progress(f"Generating image {cur_prompt_idx} of {total_prompts_num}: {key}")

            resp = requests.post(f"{base_url}/sdapi/v1/txt2img", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            images_b64 = data.get("images", []) or []
            info_raw = data.get("info", "{}")
            try:
                info = json.loads(info_raw) if isinstance(info_raw, str) else (info_raw or {})
            except Exception:
                info = {}

            all_seeds = info.get("all_seeds") or ([] if seed is None else [seed])

            if not images_b64:
                tc_message(f"No images returned for key={key}", status="WARNING")
                continue

            saved_files = []
            for idx, b64 in enumerate(images_b64, start=1):
                img_bytes = _decode_base64_image(b64)
                # 단일/다중 배치 파일명 규칙 (현재 동일 파일명 정책)
                fname = f"{key}.png" if len(images_b64) == 1 else f"{key}.png"
                fpath = outdir_string / fname

                with open(fpath, "wb") as fw:
                    fw.write(img_bytes)
                saved_files.append(str(fpath))

                # 해당 인덱스의 시드(있으면)
                seed_log = None
                try:
                    seed_log = all_seeds[idx - 1] if idx - 1 < len(all_seeds) else None
                except Exception:
                    seed_log = None

                if seed_log is not None:
                    tc_message(f"Saved: {fpath} (seed={seed_log})")
                else:
                    tc_message(f"Saved: {fpath}")

            tc_message(f"Done: {key} (saved {len(saved_files)} file(s))")

        except requests.exceptions.RequestException as e:
            tc_message(f"HTTP error for {key}: {e}", status="ERROR")
        except Exception as e:
            tc_message(f"Unexpected error for {key}: {e}", status="ERROR")

    # 메모리 해제 필요 시 사용
    # try_unload_checkpoint(base_url, timeout=timeout)

if __name__ == "__main__":
    main()