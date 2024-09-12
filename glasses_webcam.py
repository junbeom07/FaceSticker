import cv2
import numpy as np
import dlib

# 얼굴 검출기 및 랜드마크 모델 초기화
detector_hog = dlib.get_frontal_face_detector()
model_path = 'shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

# 스티커 이미지 불러오기
sticker_path = 'aaaa.png'
img_king = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)  # 스티커 이미지의 알파 채널(투명도)도 고려

# 웹캠 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_show = frame.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 검출
    dlib_rects = detector_hog(img_rgb, 1)

    for dlib_rect in dlib_rects:
        # 랜드마크 검출
        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))

        # 눈 위치 (36번 랜드마크: 왼쪽 눈의 왼쪽 끝, 45번 랜드마크: 오른쪽 눈의 오른쪽 끝)
        left_eye_x = list_points[36][0]
        left_eye_y = list_points[36][1]
        right_eye_x = list_points[45][0]
        right_eye_y = list_points[45][1]
        
        # 눈의 중심을 계산하여 스티커 위치 결정
        eye_center_x = (left_eye_x + right_eye_x) // 2
        eye_center_y = (left_eye_y + right_eye_y) // 2
        
        # 스티커의 크기 설정 (가로 크기를 더 늘림)
        w = int((right_eye_x - left_eye_x) * 1.7)  # 가로 크기 2배
        h = w // 3  # 비율에 따라 높이 조정
        
        # 스티커 이미지 리사이즈
        img_king_resized = cv2.resize(img_king, (w, h))
        
        # 스티커 위치 보정 (약간 위로 이동)
        refined_x = eye_center_x - w // 2
        refined_y = eye_center_y - h // 2 - int(h * 0.3)  # 위로 30% 정도 이동

        # 스티커가 이미지 경계 밖으로 나가지 않도록 조정
        end_x = min(refined_x + img_king_resized.shape[1], img_show.shape[1])
        end_y = min(refined_y + img_king_resized.shape[0], img_show.shape[0])

        # 영역 조정
        refined_x = max(refined_x, 0)
        refined_y = max(refined_y, 0)
        img_king_resized = img_king_resized[:end_y-refined_y, :end_x-refined_x]

        # 합성할 영역 지정
        king_area = img_show[refined_y:end_y, refined_x:end_x]

        # 스티커의 알파 채널(투명도) 처리
        if img_king_resized.shape[2] == 4:  # 알파 채널이 존재하는 경우
            alpha_mask = img_king_resized[:, :, 3] / 255.0
            for c in range(0, 3):
                king_area[:, :, c] = (1.0 - alpha_mask) * king_area[:, :, c] + alpha_mask * img_king_resized[:, :, c]
        else:  # 알파 채널이 없는 경우
            img_show[refined_y:end_y, refined_x:end_x] = np.where(img_king_resized == 0, king_area, img_king_resized).astype(np.uint8)

    # 결과 출력
    cv2.imshow('Sticker Filter', img_show)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
