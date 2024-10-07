from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize Flask
app = Flask(__name__)

# Initialize Mediapipe and deque objects for drawing
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Color points deque (deque helps in storing a sequence of coordinates)
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes for the points
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel for dilation
kernel = np.ones((5,5),np.uint8)

# Colors for drawing
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Paint Window Setup
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Capture video frames
def gen_frames():
    cap = cv2.VideoCapture(0)
    global blue_index, green_index, red_index, yellow_index, bpoints, gpoints, rpoints, ypoints, colorIndex

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            result = hands.process(framergb)

            # Post-processing the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * 640)
                        lmy = int(lm.y * 480)
                        landmarks.append([lmx, lmy])

                    # Draw landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                fore_finger = (landmarks[8][0], landmarks[8][1])
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

                # Gesture recognition for drawing
                if thumb[1] - fore_finger[1] < 30:
                    if 40 <= fore_finger[0] <= 140:
                        # Clear the screen
                        bpoints = [deque(maxlen=1024)]
                        gpoints = [deque(maxlen=1024)]
                        rpoints = [deque(maxlen=1024)]
                        ypoints = [deque(maxlen=1024)]
                        blue_index = green_index = red_index = yellow_index = 0
                        paintWindow[67:, :, :] = 255

                    # Color selection
                    elif 160 <= fore_finger[0] <= 255:
                        colorIndex = 0
                    elif 275 <= fore_finger[0] <= 370:
                        colorIndex = 1
                    elif 390 <= fore_finger[0] <= 485:
                        colorIndex = 2
                    elif 505 <= fore_finger[0] <= 600:
                        colorIndex = 3
                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(fore_finger)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(fore_finger)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(fore_finger)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(fore_finger)

            # Draw lines on the frame
            points = [bpoints, gpoints, rpoints, ypoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
