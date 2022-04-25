import sys
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import settings_2

def detect_face(model, cascade_filepath, image):
    # 이미지를 BGR형식에서 RGB형식으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(imaåge)
    # plt.show()
    # print(image.shape)
    # 그레이스케일 이미지로 변환
    image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 얼굴인식 실행
    cascade = cv2.CascadeClassifier(cascade_filepath)
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=2, minSize=(64, 64))

    # 얼굴이 1개 이상 검출된 경우
    if len(face_list) > 0:
        print(f"인식 한 얼굴의 수:{len(face_list)}")
        for (xpos, ypos, width, height) in face_list:
            # 인식된 얼굴 자르기
            face_image = image[ypos:ypos+height, xpos:xpos+width]
            print(f"인식한 얼굴의 사이즈:{face_image.shape}")
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                print("인식한 얼굴의 사이즈가 너무 작습니다.")
                continue
            # 인식한 얼굴의 사이즈 축소
            face_image = cv2.resize(face_image, (64, 64))
            # plt.imshow(face_image)
            # plt.show()
            # print(face_image.shape)
            # 인식한 얼굴 주변에 붉은색 사각형을 표시
            cv2.rectangle(image, (xpos, ypos), (xpos+width, ypos+height),
                          (255, 0, 0), thickness=2)
            # 인식한 얼굴을 1장의 사진으로 합친 배열로 변환
            face_image = np.expand_dims(face_image, axis=0)
            # 인식한 얼굴로부터 이름을 가져오기
            name = detect_who(model, face_image)
            # 인식한 얼굴에 이름 그리기
            cv2.putText(image, name, (xpos, ypos+height+20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    # 얼굴이 검출되지 않은 경우
    else:
        print(f"얼굴을 인식할 수 없습니다.")
    return image

def detect_who(model, face_image):
    # 예측
    name = ""
    result = model.predict(face_image)
    print(f"전지현일 가능성:{result[0][0]*100:.3f}%")
    print(f"송혜교일 가능성:{result[0][1]*100:.3f}%")
    name_number_label = np.argmax(result)
    if name_number_label == 0:
        name = "Jihyun"
    elif name_number_label == 1:
        name = "Hyegyo"
    return name

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
    argvs = sys.argv
    if len(argvs) != 2 or not os.path.exists(argvs[1]):
        print("이미지 파일을 지정해 주세요.")
        return RETURN_FAILURE
    image_file_path = argvs[1]

    # 이미지 파일 읽기
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"이미지 파일을 읽을 수 없습니다. {image_file_path}")
        return RETURN_FAILURE

    # 모델 파일 읽기
    if not os.path.exists(INPUT_MODEL_PATH):
        print("모델 파일이 존재하지 않습니다.")
        return RETURN_FAILURE
    model = keras.models.load_model(INPUT_MODEL_PATH)

    # 얼굴인식
    cascade_filepath = settings_2.CASCADE_FILE_PATH
    result_image = detect_face(model, cascade_filepath, image)
    plt.imshow(result_image)
    plt.show()

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()