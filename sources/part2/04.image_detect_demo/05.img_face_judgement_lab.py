import sys
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import settings_2

def detect_face(model, cascade_filepath, image):
    # 이미지를 BGR형식에서 RGB형식으로 변환
    # TO-DO
    
    # plt.imshow(image)
    # plt.show()
    # print(image.shape)

    # 그레이스케일 이미지로 변환
    # TO-DO
    # 얼굴인식 실행
    # TO-DO

    # 얼굴이 1개 이상 검출된 경우
    # TO-DO
    # 얼굴이 검출되지 않은 경우
    # TO-DO

    return image

def detect_who(model, face_image):
    # 예측
    # TO-DO

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Inpou Model Directory
INPUT_MODEL_PATH = "./model/model.h5"

def main():
    print("===================================================================")
    print("Keras를 이용한 얼굴인식")
    print("학습 모델과 지정한 이미지 파일을 기본으로 연예인 구분하기")
    print("===================================================================")

    # 인수 체크
    # TO-DO

    # 이미지 파일 읽기
    # TO-DO

    # 모델 파일 읽기
    # TO-DO

    # 얼굴인식
    # TO-DO

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()