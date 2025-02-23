from docling.document_converter import DocumentConverter

# 입력 파일
source = "test.pdf"  # 변환할 PDF 파일 경로
output_path = "output.md"  # 저장할 마크다운 파일 경로

# DocumentConverter를 사용해 변환
converter = DocumentConverter()
result = converter.convert(source)

# 결과 문자열 가져오기
converted_text = result.document.export_to_markdown()

# 결과를 마크다운 파일로 저장
with open(output_path, 'w', encoding='utf-8') as md_file:
    md_file.write(converted_text)

print(f"마크다운 문서가 {output_path}에 저장되었습니다.")
