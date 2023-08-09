import cv2
import numpy as np
import dlib
import RPi.GPIO as GPIO
import time

# GPIO setup for the alarm
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
alarm_on = False

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download this file from dlib's website
predictor = dlib.shape_predictor(predictor_path)

# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to check if the driver's eyes are closed
def are_eyes_closed(shape):
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    left_eye_avg = np.mean(left_eye, axis=0)
    right_eye_avg = np.mean(right_eye, axis=0)
    left_eye_aspect_ratio = (euclidean_distance(left_eye[1], left_eye[5]) + euclidean_distance(left_eye[2], left_eye[4])) / (2 * euclidean_distance(left_eye[0], left_eye[3]))
    right_eye_aspect_ratio = (euclidean_distance(right_eye[1], right_eye[5]) + euclidean_distance(right_eye[2], right_eye[4])) / (2 * euclidean_distance(right_eye[0], right_eye[3]))
    avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

    if avg_eye_aspect_ratio < 0.2:  # You may need to adjust this threshold based on testing
        return True
    else:
        return False

# Capture video from the Pi camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        if are_eyes_closed(shape):
            if not alarm_on:
                GPIO.output(11, GPIO.HIGH)  # Turn on the alarm
                alarm_on = True
        else:
            if alarm_on:
                GPIO.output(11, GPIO.LOW)  # Turn off the alarm
                alarm_on = False

        # Draw landmarks on the frame for visualization (optional)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
GPIO.output(11, GPIO.LOW)
GPIO.cleanup()
camera.release()
cv2.destroyAllWindows()
