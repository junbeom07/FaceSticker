import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib

image_path = "new.jpg"
img_bgr = cv2.imread(image_path)
img_show = img_bgr.copy()			
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

detector_hog = dlib.get_frontal_face_detector()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)

print(dlib_rects)   

for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

model_path = 'shape_predictor_68_face_landmarks.dat' 
landmark_predictor = dlib.shape_predictor(model_path)

list_landmarks = []

# 얼굴 영역 박스 마다 face landmark를 찾아낸다
for dlib_rect in dlib_rects:
	points = landmark_predictor(img_rgb, dlib_rect)
	# face landmark 좌표를 저장
	list_points = list(map(lambda p: (p.x, p.y), points.parts()))
	list_landmarks.append(list_points)
    
print(len(list_landmarks[0]))

for landmark in list_landmarks:
	for point in landmark:
		cv2.circle(img_show, point, 2, (255, 255, 255), -1)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()