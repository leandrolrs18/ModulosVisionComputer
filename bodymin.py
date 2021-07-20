import cv2 
import mediapipe as mp
import time

class poseDetector ():
    def __init__(self, mode =false, upBody = False, smooth = True, 
                detectionCon = 0.5, trackCon = 0.5):
        #inicialmente ao chamar essa classe, essas variaveis é criada pro usuario. Por exemplo, ao chamar a handDetector
        # mp.solutions.hands já criado
        # n posso ultilizar o parametro direto, primeiro é criado variaveis self.x com o parametro que pode ser usadas dentro da função
        # como se estivesse fazendo uma cópia, 
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose= mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody,
                                       self.smooth, self.detectionCon, self.trackCon)                    #objeto/classe que detecta mãos
        self.mpDraw = mp.solutions.drawing_utils                                            #objeto/classe que desenha a classe self.hands  

    def findPose (self, img, draw=True):
        imgRGB = cv2.cvtColor (img, cv2.COLOR_BRG2RGB)                                      #convertendo para RGB
        results = self.pose.process (imgRGB)                                               # nessa acredito que pega a posição?

        if results.pose_landmarks:                                                    # para cada mão achada
            #for handLms in results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks,                                #desenha na img original as conexões dos pontos achados
                                            self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition (self, img, draw=True):

        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate (self.results.pose_landmarks.landmark):                                      #para cada mão achada pegar o id e lm dessa posição
                h, w, c = img.shape                                                         
                cx, cy = int (lm.x*w), int(lm.y*h)                                          # o valor do lm é em 0.8 etc, precisa da conversão
                lm.List.append([id, cx, cy])                                    
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)                # desenha um circulo 
        return lmList