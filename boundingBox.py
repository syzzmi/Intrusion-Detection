import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from shapely.geometry import Point, Polygon

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video = cv2.VideoCapture('test 6.mp4')

width = 1280
height = 720

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_bb.mp4', fourcc, 20.0, (width, height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        rc, image = video.read()
        if type(image) == type(None):
            print("no video")
            break
        image = cv2.resize(image, (width, height))
        resultImage = image.copy()
        coord = [(403, 281), (389, 403), (1022, 411), (783, 293)]
        poly = Polygon(coord)

        pts = np.array(coord)
        pts = np.reshape(pts, (-1, 1, 2))
        isclose = True
        color = (0, 0, 0)
        thick = 2
        imgpoly = cv2.polylines(resultImage, [pts], isclose, color, thick)

        gray = cv2.cvtColor(imgpoly, cv2.COLOR_BGR2GRAY)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                if poly.contains(Point(x, y)):
                    cv2.circle(imgpoly, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(imgpoly, 'Warning: Restricted Area!', (144, 544), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    cv2.putText(imgpoly, 'Area Breach!', (1109, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    print('Alert: Area Breach')

                    mp_drawing.draw_landmarks(
                        imgpoly, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    # Extract bounding box coordinates
                    bboxes = []
                    for landmark in results.pose_landmarks.landmark:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        bboxes.append((x, y))

                    # Draw bounding box
                    if bboxes:
                        xmin = min(x for x, y in bboxes)
                        ymin = min(y for x, y in bboxes)
                        xmax = max(x for x, y in bboxes)
                        ymax = max(y for x, y in bboxes)
                        cv2.rectangle(imgpoly, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        out.write(imgpoly)  # Write the frame to the output video

        cv2.imshow('window', imgpoly)
        if cv2.waitKey(1) == ord('q'):
            break

# Release the VideoWriter and VideoCapture objects
out.release()
video.release()
cv2.destroyAllWindows()
