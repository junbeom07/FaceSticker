# FaceSticker

설치해야할것
_________________________________________________

pip install opencv-python

pip install numpy

pip install -U matplotlib

pip inatall dlib
_________________________________________________

https://velog.io/@yimethan/Windows10%EC%97%90-CMake-dlib-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0-Python-3.10 (dlib 설치)

vscode에서 실행시 

Ctrl+Shift+P

Python: Select Interpreter

3.12 아래 버전 사용해야 dlib설치가능

https://pixlr.com/kr/editor/
png 파일로 변경
_________________________________________________

box.py

얼굴을 검출하고, 검출된 얼굴에 사각형을 그려 나타냄

landmark.py

dlib 라이브러리를 사용하여 얼굴을 검출하고, 각 얼굴의 얼굴 랜드마크를 식별하여 나타냄

sticker.py

얼굴의 랜드마크를 기반으로 이미지에 스티커를 붙임

hat.py

얼굴 랜드마크의 머리 쪽에 스티커 이미지 = 모자

glasses.py

얼굴 랜드마크의 눈쪽에 스티커 이미지 = 안경

hat_webcam.py

웹캠 연결

glasses_webcam.py

웹캠 연결

main.py

개발중인 파일 
(s키를 눌렀을때 이미지 저장하는기능 까지 만듬)




 
