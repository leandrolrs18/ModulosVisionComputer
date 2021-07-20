import cv2
import mediapipe as mp
import time 
import bodymin as bm 

pTime= 0 
cTime = 0
cap = cv2.VideoCapture('PoseVideos/1.mp4')
#cap = cv2.VideoCapture(0)
detector = bm.poseDetector()
while True:
    sucess, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img, draw=True)
    if len(lmList) != 0:                                                    #só mostra se mão aparecer na imagem 
        print(lmList[14])

    cTime = time.time()
    fps = 1/ (cTime- pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN, 3, 
                (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)