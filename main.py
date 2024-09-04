import numpy as np
import cv2
import mediapipe as mp
import pyglet

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize MediaPipe Hands model
with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    keys = [["C", "D", 'E', "F", "G", "A", "B", "C", "D", "E", "F", "G", "A", "B"], 
            ["C#", "D#", "F#", "G#", "A#", "C#", "D#", "F#", "G#", "A#"]]

    class Button():
        def __init__(self, pos, text, size, color):
            self.pos = pos
            self.size = size
            self.text = text
            self.color = color

    buttonList = []
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):   
            if i == 0:
                buttonList.append(Button([38 * j + 15, 80], key, [35, 100], (255, 255, 255)))
            else:
                buttonList.append(Button([40 * j + 25, 80], key, [35, 50], (0, 0, 0)))    

    def playkeys(button):
        sound_path = button.text + ".wav"  # Adjust according to your naming convention
        try:
            effect = pyglet.resource.media(sound_path, streaming=False)
            effect.play()
        except Exception as e:
            print(f"Error playing {sound_path}: {e}")

    def drawAll(img, buttonList):
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            color = button.color
            cv2.rectangle(img, button.pos, (x + w, y + h), color, cv2.FILLED)
            cv2.putText(img, button.text, (x + 10, y + h - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (214, 0, 220), 2)
        return img    

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        # Convert the BGR image to RGB before processing.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Draw the button interface
        img = drawAll(img, buttonList)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Example: Getting the position of a specific landmark (index finger tip)
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    # Checking fingertip positions (e.g., index finger tip: landmark 8)
                    fingertip = hand_landmarks.landmark[8]  # Index finger tip
                    h, w, _ = img.shape
                    fingertip_x, fingertip_y = int(fingertip.x * w), int(fingertip.y * h)

                    # Check if fingertip is within button bounds
                    if x < fingertip_x < x + w and y < fingertip_y < y + h:
                        playkeys(button)

        cv2.imshow("IMAGE", img)

        # Add exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Proper cleanup
    cap.release()
    cv2.destroyAllWindows()

# import numpy as np
# import time
# import cv2
# import mediapipe as mp
# # from cvzone.HandTrackingModule import HandDetector
# import pyglet


# cap =cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# window = pyglet.window.Window()
# # detector =HandDetector(detectionCon=0.8)
# detector = handDetector(detectionCon=0.8)

# keys=[["C","D",'E',"F","G","A","B","C","D","E","F","G","A","B"],["C#","D#","F#","G#","A#","C#","D#","F#","G#","A#"]]

# class Button():
#     def __init__(self,pos,text,size,color):
#         self.pos=pos
#         self.size=size
#         self.text=text
#         self.color=color
# buttonList=[]
# for i in range(len(keys)):
#     for j,key in enumerate(keys[i]):   
#         if i==0:
#             buttonList.append(Button([38*j+15,80],key,[35,100],(255,255,255)))
#         else:
#             buttonList.append(Button([(40+j)*j+25,80],key,[35,50],(0,0,0)))    

# def playkeys(button):
#     if button.text=="A":
            
#         effectA=pyglet.resource.media("A.wav",streaming=False)
#         effectA.play()
                
                
#     elif button.text=="B":
            
#         effectB=pyglet.resource.media("B.wav",streaming=False)
#         effectB.play()
                
#     elif button.text=="C":
            
#         effectC=pyglet.resource.media("C.wav",streaming=False)
#         effectC.play()
#     elif button.text=="D":
            
#         effectD=pyglet.resource.media("D.wav",streaming=False)
#         effectD.play()
#     elif button.text=="E":
            
#         effectE=pyglet.resource.media("E.wav",streaming=False)
#         effectE.play()
        

#     elif button.text=="F":
            
#         effectF=pyglet.resource.media("F.wav",streaming=False)
#         effectF.play()
#     elif button.text=="G":
            
#         effectG=pyglet.resource.media("G.wav",streaming=False)
#         effectG.play()                  


# def drawAll(img,buttonList):
#     for button in buttonList:
#         x,y=button.pos
#         w,h=button.size
#         colorr=button.color
#         cv2.rectangle(img,button.pos,(x+w,y+h),colorr,cv2.FILLED)
#         cv2.putText(img,button.text,(x+10,y+h-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(214,0,220),2)
#     return img    

# while True:
#     success,img=cap.read()



#     img= detector.findHands(img)
#     lmlist,bboxInfo=detector.findPosition(img)
#     img=drawAll(img,buttonList)
#     if lmlist:        #hand is there
#         for button in buttonList:
#             x,y=button.pos
#             w,h=button.size
            
#             for f in [4,8,12,16,20]:

#                  if x<lmlist[f][0]<x+w and y<lmlist[f][1]<y+h:
#                      l,_,_=detector.findDistance(f,f-3,img,draw=False)
#                      if l<120:
#                          #cv2.rectangle(img,button.pos,(x+w,y+h),(80,9,78),cv2.FILLED)
#                          playkeys(button)

                         

                     



    


#     cv2.imshow("IMAGE",img)
#     cv2.waitKey(1)
   



