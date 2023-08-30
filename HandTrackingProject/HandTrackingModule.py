import cv2
import mediapipe as mp
import time


class HandDetector():

    def __init__(self, mode=False, maxHands=2):
        
        self.mode = mode
        self.maxHands = maxHands

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB)

        ## Detection de présence de main
        ##print(result.multi_hand_landmarks)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)         
        return img
                
 

def main() :

    ## Current Time
    cTime = 0
    ## Previous Time
    pTime = 0

    ## Hand Tracking
    cap = cv2.VideoCapture(1)

    detector = HandDetector()
    

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        ## Calcul du FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        ## Affichage du FPS
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1)

    

if __name__ == "__main__":
    main()