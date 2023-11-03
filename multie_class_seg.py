import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
import cv2

# 모델 경로 지정
model_path = './selfie_multiclass_256x256.tflite'

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ImageSegmenter 옵션 정의
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, # 작업의 실행 모드 설정 (IMAGE, VIDEO, LIVE_STREAM)
    output_category_mask=True) #Tru로 설정될 경우 분할 마스크가 uint8로 포함됨.

# 색상 정의 (BGR 형식)
colors = [
    [255, 0, 0],  # background: blue
    [255, 255, 0],  # hair: skyblue
    [0, 255, 0],  # body-skin: green
    [255, 0, 128],  # face-skin: purple
    [255, 0, 255],  # clothes: magenta
    [0, 128, 255]  # others: orange
]

# 카테고리 정의
category = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]


with ImageSegmenter.create_from_options(options) as segmenter:

    # 이미지 불러오기
    # mp_image = mp.Image.create_from_file('./image/sampl3.jpg')
    mp_image = mp.Image.create_from_file('./image/sample2.png')
    # mp_image = mp.Image.create_from_file('./image/sample.jpg')

    # 원본 이미지 가져오기 (알파 채널 제거)
    original_image = mp_image.numpy_view()
    original_image = original_image[:, :, :3]  # 알파 채널 제거 -> addWeight를 위한 작업

    # 이미지 분할
    segmented_masks = segmenter.segment(mp_image)
    print(segmented_masks)

    # 결과값에서 category_mask 속성 가져오기
    category_mask = segmented_masks.category_mask
    category_mask_np = category_mask.numpy_view()

    # 컬러 이미지 생성
    h, w = category_mask_np.shape # numpy 배열의 width, height 사이즈 가져옴
    color_image = np.zeros((h, w, 3), dtype=np.uint8) #동일한 크기의 빈 이미지 생성
    for i, color in enumerate(colors): # 각 인덱스(카테고리)에 대한 색상 채워넣기
        color_image[category_mask_np == i] = color

    # 각 카테고리 존재 여부 확인
    for i in range(len(category)):
        is_present = np.isin(category_mask_np, i).any()
        print(f"{category[i]} 존재 : {is_present}")


    # 이미지 투명도 설정
    alpha = 0.5
    blended_image = cv2.addWeighted(original_image, 1 - alpha, color_image, alpha, 0)

    #결과
    cv2.imshow('Color', color_image)
    cv2.imshow('Original_color', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
