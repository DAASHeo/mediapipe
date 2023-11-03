import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
import matplotlib.pyplot as plt
from mediapipe.tasks.python import vision
import cv2

model_path = './selfie_multiclass_256x256.tflite'
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a image segmenter instance with the live stream mode:
def print_result(result: List[Image], output_image: Image, timestamp_ms: int):
    print('segmented masks size: {}'.format(len(result)))

options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.VIDEO,
    output_category_mask=True,
    output_callback=print_result  # 콜백 함수 설정
)

colors = [
    [0, 0, 0],  # background: black
    [0, 0, 255],  # hair: red
    [0, 255, 0],  # body-skin: green
    [255, 0, 0],  # face-skin: blue
    [255, 255, 225],  # clothes:white
    [255, 0, 255]  # others: magenta
]

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 확인: 카메라가 제대로 열렸는지 확인
if not cap.isOpened():
    print("Cannot open camera")
    exit()


with ImageSegmenter.create_from_options(options) as segmenter:
    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            print("Failed to grab frame")
            break

        # OpenCV 이미지를 Mediapipe 이미지로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # 분할 수행
        segmented_masks = segmenter.segment(mp_image)
        category_mask = segmented_masks.category_mask

        # NumPy 배열로 변환
        category_mask_np = category_mask.numpy_view()

        # 컬러 이미지 생성
        h, w = category_mask_np.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            color_image[category_mask_np == i] = color

        # 컬러 이미지 표시
        cv2.imshow('Color Image', color_image)
        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 눌러 종료
            break

    # 종료 시 웹캠 및 OpenCV 창 닫기
    cap.release()
    cv2.destroyAllWindows()
