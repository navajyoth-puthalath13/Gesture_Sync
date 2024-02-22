import time
import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=1, detection_confidence=0.5, tracking_confidence=0.5):  # Constructor to initialize the hand detector object
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        # Initialize MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, int(self.maxHands), int(self.detection_confidence),int(self.tracking_confidence))
        self.mpDraw = mp.solutions.drawing_utils
        self.tipId = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):            # Function to detect hands in the image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if xList and yList:
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingerUp(self):
        fingers = []
        if len(self.lmList) != 0:
            if self.lmList[self.tipId[0]][1] > self.lmList[self.tipId[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if self.lmList[self.tipId[id]][1] > self.lmList[self.tipId[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers


def main():
    # cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)       # Open default camera
    detector = handDetector()       # Create an instance of handDetector class

    while True:
        success, img = cap.read()  # Read frame from the camera
        img = detector.findHands(img)  # Detect hands in the frame
        lmList = detector.findPosition(img)  # Find landmark positions
        if len(lmList) != 0:
            print(lmList)        # Print the position of the 5th landmark (for example)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('video', img)        # Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()                # Release the camera
    cv2.destroyAllWindows()      # Close all OpenCV windows


if __name__ =="__main__":
    main()