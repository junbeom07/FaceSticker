import cv2
import numpy as np
import dlib
from datetime import datetime

# 이미지 경로 지정
sticker_path = 'aaaa.png'
img_gla = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)

# 모델 설정
detector_hog = dlib.get_frontal_face_detector()
model_path = 'shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

# 스티커 크기 조절 변수
scaling_factor_width = 1.0  # 기본 가로 크기 비율
scaling_factor_height = 1.0  # 기본 세로 크기 비율

# 원래 크기 저장
original_scaling_factor_width = scaling_factor_width
original_scaling_factor_height = scaling_factor_height

# 웹캠 0번으로 설정
cap = cv2.VideoCapture(0)

# 프레임이 안잡히면 종료
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
        
        # 스티커의 크기 설정
        w = int((right_eye_x - left_eye_x) * 1.7 * scaling_factor_width)  # 가로 크기 조절
        h = int(w // 3 * scaling_factor_height)  # 비율에 따라 높이 조정
        
        # 스티커 이미지 리사이즈
        img_gla_resized = cv2.resize(img_gla, (w, h))
        
        # 스티커 위치 보정 (약간 위로 이동)
        refined_x = eye_center_x - w // 2
        refined_y = eye_center_y - h // 2 - int(h * 0.3)  # 위로 30% 정도 이동

        # 스티커가 이미지 경계 밖으로 나가지 않도록 조정
        end_x = min(refined_x + img_gla_resized.shape[1], img_show.shape[1])
        end_y = min(refined_y + img_gla_resized.shape[0], img_show.shape[0])

        # 영역 조정
        refined_x = max(refined_x, 0)
        refined_y = max(refined_y, 0)
        img_gla_resized = img_gla_resized[:end_y-refined_y, :end_x-refined_x]

        # 합성할 영역 지정
        king_area = img_show[refined_y:end_y, refined_x:end_x]

        # 스티커의 알파 채널(투명도) 처리
        if img_gla_resized.shape[2] == 4:  # 알파 채널이 존재하는 경우
            alpha_mask = img_gla_resized[:, :, 3] / 255.0
            for c in range(0, 3):
                king_area[:, :, c] = (1.0 - alpha_mask) * king_area[:, :, c] + alpha_mask * img_gla_resized[:, :, c]
        else:  # 알파 채널이 없는 경우
            img_show[refined_y:end_y, refined_x:end_x] = np.where(img_gla_resized == 0, king_area, img_gla_resized).astype(np.uint8)

    # 결과 출력
    cv2.imshow('Sticker Filter', img_show)
    
    # 사용자 입력을 통해 스티커 크기 조절 및 캡처 기능
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):
        scaling_factor_width = max(0.1, scaling_factor_width - 0.1)  # 가로 크기 감소
    elif key == ord('d'):
        scaling_factor_width += 0.1  # 가로 크기 증가
    elif key == ord('w'):
        scaling_factor_height += 0.1  # 세로 크기 증가
    elif key == ord('s'):
        scaling_factor_height = max(0.1, scaling_factor_height - 0.1)  # 세로 크기 감소
    elif key == ord('m'):
        # 현재 프레임 캡처
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'captured_frame_{timestamp}.png'
        cv2.imwrite(filename, img_show)
        print(f"Frame captured as '{filename}'")
    elif key == ord('f'):
        # 원래 크기로 리셋
        scaling_factor_width = original_scaling_factor_width
        scaling_factor_height = original_scaling_factor_height

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
