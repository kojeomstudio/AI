import easyocr
import cv2

# EasyOCR 리더 초기화 (한글 + 영어)
reader = easyocr.Reader(['ko', 'en'], gpu=True)

def extract_text_from_image(bgr_image) -> str:
    import cv2

    # 밝기 채널 강화
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)

    # 적응형 threshold로 텍스트 대비 강화
    thresh = cv2.adaptiveThreshold(
        eq, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 8
    )

    # EasyOCR은 RGB 사용 → binary threshold 대신 원본 RGB도 함께 시도 가능
    from easyocr import Reader
    reader = Reader(['ko', 'en'], gpu=False)
    result = reader.readtext(thresh, detail=0, paragraph=False)
    return " ".join(result).strip()

