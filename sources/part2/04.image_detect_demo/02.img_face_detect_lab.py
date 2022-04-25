#-*- coding: utf-8 -*- 
import os
import pathlib
import glob
import cv2
import settings_2

def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 Path Pattern에 일치하는 파일 얻기
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        # 파일 경로
        # 파일명
        # 이미지 읽기
        # TO-DO 
        
    return name_images

def detect_image_face(file_path, image, cascade_filepath):
    # 이미지 파일의 Grayscale화
    # 캐스케이드 파일 읽기
    # 얼굴인식
    # TO-DO 

    # 1개 이상의 얼굴인식
    # TO-DO 

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
# Origin Image Pattern
IMAGE_PATH_PATTERN = "./origin_image/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./face_image"

def main():
    print("===================================================================")
    print("이미지 얼굴인식 OpenCV 이용")
    print("지정한 이미지 파일의 정면얼굴을 인식하고, 64x64 사이즈로 변경")
    print("===================================================================")

    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내의 파일 제거
    delete_dir(OUTPUT_IMAGE_DIR, False)

    # 이미지 파일 읽기
    # TO-DO 

    # 이미지별로 얼굴인식
    for name_image in name_images:
        # TO-DO 

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()