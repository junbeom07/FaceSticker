import cv2
import numpy as np
import dlib

# 얼굴 검출기와 랜드마크 모델 초기화
detector_hog = dlib.get_frontal_face_detector()
model_path = 'shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

# 스티커 이미지 불러오기
sticker_path = 'kkkk.png'
img_king = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)  # 투명한 부분까지 고려해 불러오기

# 웹캠 초기화
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_show = frame.copy()
    
    # 얼굴 검출
    dlib_rects = detector_hog(img_rgb, 1)
    
    list_landmarks = []

    # 얼굴마다 랜드마크 검출
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)

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
        end_x = min(refined_x + img_king_resized.shape[1], img_show.shape[1])
        end_y = min(refined_y + img_king_resized.shape[0], img_show.shape[0])
        
        # 영역 조정
        img_king_resized = img_king_resized[:end_y-refined_y, :end_x-refined_x]
        refined_y = max(refined_y, 0)
        refined_x = max(refined_x, 0)

        # 합성할 영역 지정
        king_area = img_show[refined_y:end_y, refined_x:end_x]

        # 알파 채널 처리하여 합성
        alpha_s = img_king_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            king_area[:, :, c] = (alpha_s * img_king_resized[:, :, c] +
                                  alpha_l * king_area[:, :, c])
        
        img_show[refined_y:end_y, refined_x:end_x] = king_area

    # 결과 프레임 출력
    cv2.imshow('Webcam', img_show)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠과 윈도우 해제
cap.release()
cv2.destroyAllWindows()
