# remove_bg.py (튜닝 버전)
import sys
from pathlib import Path

from rembg import remove, new_session
from PIL import Image

def main():
    if len(sys.argv) < 2:
        print("사용법: python remove_bg.py <이미지_폴더_경로> [model_name]")
        print("model_name 예: isnet-general-use | u2net | u2net_human_seg | isnet-anime | u2netp")
        sys.exit(1)

    img_dir = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) >= 3 else "isnet-general-use"

    if not img_dir.is_dir():
        print(f"폴더가 아닙니다: {img_dir}")
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not files:
        print("처리할 이미지가 없습니다.")
        return

    session = new_session(model_name)

    for src in files:
        if "rm" in src.stem:
            continue

        try:
            with Image.open(src) as im:
                im = im.convert("RGBA")
                out_im = remove(
                    im,
                    session=session,
                    post_process=True,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,  # 210~260 사이로 조정해보세요
                    alpha_matting_background_threshold=10,    # 5~50 사이로 조정
                    alpha_matting_erode_size=5                # 0~20 사이로 조정
                )
                out_path = src.with_name(f"{src.stem}_rm.png")
                out_im.save(out_path)
                print(f"OK: {src.name} -> {out_path.name}")
        except Exception as e:
            print(f"FAIL: {src.name} ({e})")

if __name__ == "__main__":
    main()