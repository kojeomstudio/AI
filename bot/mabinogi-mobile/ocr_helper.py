import cv2

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        from easyocr import Reader
        _reader = Reader(['ko', 'en'], gpu=True)
    return _reader

def extract_text_from_image(bgr_image) -> str:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(
        eq, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 8
    )
    reader = _get_reader()
    result = reader.readtext(thresh, detail=0, paragraph=False)
    return " ".join(result).strip()
