# Import necessary libraries
import cv2
import mediapipe as mp
import time
from collections import deque
import numpy as np

# Function to automatically adjust brightness and contrast
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram of the grayscale image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Compute cumulative distribution (accumulator)
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    max_value = accumulator[-1]
    clip_hist_percent *= (max_value / 100.0)
    clip_hist_percent /= 2.0

    # Find minimum and maximum gray levels to clip
    min_gray = 0
    while accumulator[min_gray] < clip_hist_percent:
        min_gray += 1

    max_gray = hist_size - 1
    while accumulator[max_gray] >= (max_value - clip_hist_percent):
        max_gray -= 1

    # Compute alpha (contrast) and beta (brightness)
    if max_gray - min_gray == 0:
        alpha = 1.0
    else:
        alpha = 255 / (max_gray - min_gray)
    beta = -min_gray * alpha

    # Apply contrast and brightness adjustments
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks indices for MediaPipe FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    # Compute distances between vertical and horizontal eye landmarks
    A = ((eye_landmarks[1][0] - eye_landmarks[5][0]) ** 2 + (eye_landmarks[1][1] - eye_landmarks[5][1]) ** 2) ** 0.5
    B = ((eye_landmarks[2][0] - eye_landmarks[4][0]) ** 2 + (eye_landmarks[2][1] - eye_landmarks[4][1]) ** 2) ** 0.5
    C = ((eye_landmarks[0][0] - eye_landmarks[3][0]) ** 2 + (eye_landmarks[0][1] - eye_landmarks[3][1]) ** 2) ** 0.5
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds and parameters
EAR_THRESHOLD = 0.25             # EAR below this indicates closed eyes
CONSEC_FRAMES = 3                # Number of frames to confirm a blink
CLOSED_EYE_DURATION = 1.5        # Duration (in seconds) to detect drowsiness
SENSITIVITY = 0.5                # Drowsiness sensitivity multiplier

# Initialize counters and states
blink_count = 0
frame_counter = 0
closed_eye_start_time = None
blink_timestamps = deque()      # Track blink times (for frequency)
drowsiness_count = 0
drowsiness_active = False

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Automatically adjust brightness and contrast
    frame = automatic_brightness_and_contrast(frame)

    # Convert frame to RGB for MediaPipe processing
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Check if face was detected
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]

        # Extract eye landmarks from full facial landmarks
        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw green dots around the eyes
        for x, y in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Check if eyes are closed
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if closed_eye_start_time is None:
                closed_eye_start_time = time.time()
            else:
                elapsed_time = time.time() - closed_eye_start_time
                if elapsed_time >= CLOSED_EYE_DURATION * SENSITIVITY:
                    # Display drowsiness warning
                    text = "Drowsiness detected 😴"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    x_pos = width - text_size[0] - 50
                    y_pos = 100
                    cv2.putText(frame, text, (x_pos, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    # Count drowsiness event once
                    if not drowsiness_active:
                        drowsiness_count += 1
                        drowsiness_active = True
        else:
            # If blink was valid (short eye closure)
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
                blink_timestamps.append(time.time())
            # Reset counters
            frame_counter = 0
            closed_eye_start_time = None
            drowsiness_active = False

        # Remove old blinks (older than 60s)
        current_time = time.time()
        while blink_timestamps and current_time - blink_timestamps[0] > 60:
            blink_timestamps.popleft()
        blink_frequency = len(blink_timestamps)

        # Display blink and drowsiness information
        cv2.putText(frame, f"Blinks: {blink_count}", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f"Frequency: {blink_frequency} per minute", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f"Drowsiness events: {drowsiness_count}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    else:
        # If no face is detected
        cv2.putText(frame, "No face detected", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Fatigue Detection", frame)

    # Exit on pressing ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
