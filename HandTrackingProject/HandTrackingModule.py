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
        self.result = self.hands.process(imgRGB)

        ## Detection de présence de main
        ##print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)         
        
        return img



    def findPosition(self, img, handNum=0, draw=True):

        lmList = []

        if self.result.multi_hand_landmarks:

            myHand = self.result.multi_hand_landmarks[handNum]

            # Éxtraction des points de repère
            for id, lm in enumerate(myHand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                
                #print(id, cx, cy)
                lmList.append([id,cx,cy])

                if draw :
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
                
        return lmList
                
 

def main() :

    ## Current Time
    cTime = 0
    ## Previous Time
    pTime = 0

    ## Hand Tracking
    cap = cv2.VideoCapture(0)

    detector = HandDetector()
    

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList = detector.findPosition(img)
        
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

    

if __name__ == "__main__":
    main()