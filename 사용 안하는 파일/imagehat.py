import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib

# 이미지 불러오기
image_path = "new.jpg"
img_bgr = cv2.imread(image_path)
img_show = img_bgr.copy()			
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 얼굴 검출기 초기화
detector_hog = dlib.get_frontal_face_detector()

# 얼굴 검출
dlib_rects = detector_hog(img_rgb, 1)

# 랜드마크 모델 초기화
model_path = 'shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

list_landmarks = []

# 얼굴마다 랜드마크 검출
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

# 스티커 이미지 불러오기
sticker_path = 'king.png'
img_king = cv2.imread(sticker_path)

# 얼굴마다 스티커 합성
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    # 코 위치 (30번 랜드마크) 기준으로 스티커 위치 계산
    x = landmark[30][0]
    y = landmark[30][1] - dlib_rect.height() // 2
    w = h = dlib_rect.width()
    
    # 스티커 이미지 리사이즈
    img_king_resized = cv2.resize(img_king, (w, h))
    
    # 스티커 위치 보정
    refined_x = x - w // 2
    refined_y = y - h
    
    if refined_x < 0:
        img_king_resized = img_king_resized[:, -refined_x:]
        refined_x = 0
    if refined_y < 0:
        img_king_resized = img_king_resized[-refined_y:, :]
        refined_y = 0

    # 스티커가 이미지 경계 밖으로 나가지 않도록 조정
    end_x = min(refined_x + img_king_resized.shape[1], img_bgr.shape[1])
    end_y = min(refined_y + img_king_resized.shape[0], img_bgr.shape[0])
    
    # 영역 조정
    img_king_resized = img_king_resized[:end_y-refined_y, :end_x-refined_x]
    refined_y = max(refined_y, 0)
    refined_x = max(refined_x, 0)

    # 합성할 영역 지정
    king_area = img_show[refined_y:end_y, refined_x:end_x]
    img_show[refined_y:end_y, refined_x:end_x] = np.where(img_king_resized == 0, king_area, img_king_resized).astype(np.uint8)

# 결과 이미지 출력
img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
