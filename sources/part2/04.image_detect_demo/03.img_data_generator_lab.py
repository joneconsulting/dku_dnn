import os
import random
import pathlib
import shutil
import glob
import cv2
import numpy as np

def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 경로에서 이미지 파일 취득
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # 파일 경로
        fullpath = str(path.resolve())
        print(f"이미지 파일(절대 경로):{fullpath}")
        # 파일명
        filename = path.name
        print(f"이미지 파일(파일명):{filename}")
        # 이미지 읽기
        image = cv2.imread(fullpath)
        if image is None:
            print(f"이미지 파일[{fullpath}]을 읽을 수가 없습니다.")
            continue
        name_images.append((filename, image))
    return name_images

def scratch_image(image, use_flip=True, use_threshold=True, use_filter=True):
    # 어떤 증가규칙을 사용할 것인지 설정 (Flip or 밝기 or 화소 ...)
    # TO-DO    
    
    # 흐린 필터 작성
    # filter1 = np.ones((3, 3))
    # 오리지날 이미지를 배열로 저장
    # TO-DO

    # 증가 규칙을 위한 함수
    # TO-DO

    # 이미지 증가
    # TO-DO

    return images

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Test Image Directory
TEST_IMAGE_PATH = "./test_image"
# Face Image Directory
IMAGE_PATH_PATTERN = "./face_image/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./face_scratch_image"

def main():
    print("===================================================================")
    print("이미지 증가를 위한 OpenCV 이용")
    print("지정한 이미지 파일의 수를 증가 (Flip, 임계값 등의 작업으로 8배 증가)")
    print("===================================================================")

    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내 파일 제거
    delete_dir(OUTPUT_IMAGE_DIR, False)

    # 디렉토리 작성
    if not os.path.isdir(TEST_IMAGE_PATH):
        os.mkdir(TEST_IMAGE_PATH)
    # 디렉토리 내 파일 제거
    delete_dir(TEST_IMAGE_PATH, False)

    # 대상 이미지에 2배 정보를 테스트용 파일로 구분
    # TO-DO

    # 대상 이미지 읽기
    name_images = load_name_images(IMAGE_PATH_PATTERN)

    # 대상 이미지 별로 증가 작업
    # for name_image in name_images:
    #     # TO-DO

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()