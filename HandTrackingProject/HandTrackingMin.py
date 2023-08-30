import cv2
import mediapipe as mp
import time

## Hand Tracking
cap = cv2.VideoCapture(1)


# Associer le module de détection de mains à un alias plus court
mpHands = mp.solutions.hands
# Créer un objet Hands pour la détection des mains
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

## Current Time
cTime = 0

## Previous Time
pTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    ## Detection de présence de main
    ##print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            ## Éxtraction des points de repère
            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)

                ## Affichage du point de repère 0
                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
    ## Calcul du FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    ## Affichage du FPS
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

    

str = """ ## Éxtraction des points de repère
                    for id, lm in enumerate(handLms.landmark):

                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        print(id, cx, cy)

                        ## Affichage du point de repère 0
                        if id == 4:
                            cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)"""
    