import time
import cv2
import mediapipe as mp
import  handmod as mm

cap = cv2.VideoCapture(0)       #captureing video form webcam

decorator=mm.handDetector
mpHands = mp.solutions.hands
hands = mpHands.Hands()       # Initialize MediaPipe Hands
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0                       #Initializes cTime and pTime

while True:
    success, img = cap.read()
    if not success:
        break                                           #read

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)                          #converting BGR to RGB

    if results.multi_hand_landmarks:                         #Checks if there are any hand landmarks detected in the frame.
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):         #iterates over each hand and draws landmarks and connections on the frame using OpenCV functions.
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), 2, cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)             #prints the coordinates of the specific landmark (id=4) to the console.

    cvr = decorator.fingerUp(imgRGB)
    cTime = time.time()
    fps = 1 / (cTime - pTime)        #Calculates the frame per second (FPS) by measuring the time taken to process the current frame.
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)             #Adds a text overlay to the frame showing the FPS value.

    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):        # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
