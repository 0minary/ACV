import cv2
import mediapipe as mp
import time

import HandTrackingModule as htm

## Current Time
cTime = 0
## Previous Time
pTime = 0

## Hand Tracking
cap = cv2.VideoCapture(0)

detector = htm.HandDetector()


while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPosition(img,draw=False)
    
    if len(lmList) != 0:
        print(lmList[4])


    ## Calcul du FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    ## Affichage du FPS
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
