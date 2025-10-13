#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
import requests

def getenv(name, default=None, cast=None):
    val = os.environ.get(name, default)
    if cast and val is not None:
        try:
            return cast(val)
        except Exception:
            return default
    return val

def main():
    # --- 1) 경로/환경 ---
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.json"

    host = getenv("SD_HOST", "127.0.0.1")
    port = getenv("SD_PORT", "7860")
    base_url = f"http://{host}:{port}"

    outdir_root = Path(getenv("OUTDIR", "out"))
    outdir_root.mkdir(parents=True, exist_ok=True)

    # --- 2) 공통 인자 (빌드스텝 환경변수) ---
    model = getenv("MODEL", "")                         # 선택
    sampler = getenv("SAMPLER", "Euler a")
    steps = getenv("STEPS", None, cast=int)            # ✅ 필수 권장(전역만 사용)
    if steps is None or steps <= 0:
        print("##teamcity[buildProblem description='env.STEPS must be a positive integer']")
        raise SystemExit(1)

    cfg_scale = getenv("CFGSCALE", "7.0", cast=float)  # ✅ 전역 CFGSCALE
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

    # --- 3) config 로드 (딕셔너리만 허용) ---
    if not config_path.exists():
        print("##teamcity[buildProblem description='config.json not found']")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    prompts = config.get("prompts", {})
    if not isinstance(prompts, dict) or not prompts:
        print("##teamcity[buildProblem description='prompts must be a non-empty object (key=name, value=prompt string)']")
        raise SystemExit(1)

    # --- 4) 프롬프트 루프 (모든 프롬프트에 동일 STEPS/CFGSCALE 적용) ---
    for key, prompt in prompts.items():
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"##teamcity[message text='Skip empty prompt for key {key}' status='WARNING']")
            continue

        outdir_string = outdir_root
        outdir_string.mkdir(parents=True, exist_ok=True)

        filename_pattern = f"{key}"

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,                  # ✅ 전역 STEPS
            "cfg_scale": cfg_scale,          # ✅ 전역 CFGSCALE
            "width": width,
            "height": height,
            "seed": seed,
            "sampler_name": sampler,
            "batch_size": batch_size,

            "save_images": True,
            "send_images": False,

            "enable_hr": hr_second_pass,
            "hr_scale": hr_upscale if hr_second_pass else 1.0,
            "hr_upscaler": hr_upscaler if hr_second_pass else None,
            "denoising_strength": denoising_strength if hr_second_pass else None,

            "override_settings": {
                **({"sd_model_checkpoint": model} if model else {}),
                "outdir_txt2img_samples": str(outdir_string),
                "samples_filename_pattern": filename_pattern,
                "save_to_dirs": False,  # ✅ 날짜 폴더 생성 금지
            },
            "override_settings_restore_afterwards": True
        }

        try:
            print(f"##teamcity[message text='Generating key={key} -> {outdir_string}']")
            resp = requests.post(f"{base_url}/sdapi/v1/txt2img", json=payload, timeout=timeout)
            resp.raise_for_status()
            print(f"##teamcity[message text='Done: {key}']")
        except requests.exceptions.RequestException as e:
            print(f"##teamcity[buildProblem description='HTTP error for {key}: {e}']")
        except Exception as e:
            print(f"##teamcity[buildProblem description='Unexpected error for {key}: {e}']")

if __name__ == "__main__":
    main()