import os
import numpy as np
from PIL import Image
import argparse

# argparse로 실행 인자를 받아 처리
def parse_args():
    parser = argparse.ArgumentParser(description="이미지 파일을 npy 파일로 변환")
    parser.add_argument('--image_dir', required=True, help='이미지 파일이 저장된 디렉토리 경로')
    parser.add_argument('--output_npy_file', required=True, help='저장할 npy 파일 경로')
    parser.add_argument('--image_size', type=int, nargs=2, default=(128, 128), help='리사이즈할 이미지 크기 (width, height)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 이미지가 저장된 디렉토리와 npy 파일을 저장할 경로
    image_dir = args.image_dir
    output_npy_file = args.output_npy_file
    image_size = args.image_size  # 리사이즈할 이미지 크기 (width, height)

    # 모든 이미지 파일을 담을 리스트 생성
    image_data = []

    # 이미지 디렉토리 내의 모든 파일을 읽음
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_dir, filename)

            # 이미지를 열고, RGBA로 변환 후 크기를 조정하여 numpy 배열로 변환
            img = Image.open(image_path).convert('RGBA')  # RGBA로 변환
            img = img.resize(image_size)  # 이미지 리사이즈
            img_array = np.array(img)

            # 이미지 배열을 리스트에 추가
            image_data.append(img_array)

    # 리스트를 numpy 배열로 변환
    image_data = np.array(image_data)

    # numpy 배열을 .npy 파일로 저장
    np.save(output_npy_file, image_data)

    print(f"이미지 데이터가 {output_npy_file}로 저장되었습니다.")

if __name__ == "__main__":
    main()
