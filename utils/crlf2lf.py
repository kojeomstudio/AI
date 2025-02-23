import os

# 변환할 텍스트 파일 확장자 목록
TEXT_EXTENSIONS = {".txt", ".md", ".html", ".css", ".js", ".ts", ".tsx", ".jsx", ".json", ".xml", ".csv", ".py", ".c", ".cpp", ".java", ".cs", ".sh", ".bat"}

# 제외할 바이너리 파일 확장자 목록 (이미지, 압축 파일 등)
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".mp3", ".mp4", ".zip", ".rar", ".tar", ".gz", ".exe", ".dll", ".bin"}

def is_text_file(file_path):
    """텍스트 파일 여부를 확인하는 함수 (확장자 기반)"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in TEXT_EXTENSIONS and ext not in BINARY_EXTENSIONS

def convert_line_endings(directory):
    modified_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            if not is_text_file(file_path):  # 이미지 및 바이너리 파일 제외
                continue

            try:
                with open(file_path, 'rb') as f:
                    content = f.read()

                new_content = content.replace(b'\r\n', b'\n')

                if new_content != content:  # 변경된 파일만 저장
                    with open(file_path, 'wb') as f:
                        f.write(new_content)
                    modified_files.append(file_path)

            except Exception as e:
                print(f"❌ 파일 처리 실패: {file_path} - {e}")

    print(f"\n✅ 변환된 파일 개수: {len(modified_files)}")
    for file in modified_files:
        print(f"✔ {file}")

if __name__ == "__main__":
    target_directory = "./R2R-Application"
    if os.path.isdir(target_directory):
        convert_line_endings(target_directory)
    else:
        print("❌ 유효한 폴더 경로를 입력하세요.")
