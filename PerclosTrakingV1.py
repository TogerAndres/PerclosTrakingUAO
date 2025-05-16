import cv2
import mediapipe as mp
import time
from collections import deque
import numpy as np

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    max_value = accumulator[-1]
    clip_hist_percent *= (max_value/100.0)
    clip_hist_percent /= 2.0

    min_gray = 0
    while accumulator[min_gray] < clip_hist_percent:
        min_gray += 1

    max_gray = hist_size -1
    while accumulator[max_gray] >= (max_value - clip_hist_percent):
        max_gray -= 1

    if max_gray - min_gray == 0:
        alpha = 1.0
    else:
        alpha = 255 / (max_gray - min_gray)
    beta = -min_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_landmarks):
    A = ((eye_landmarks[1][0] - eye_landmarks[5][0]) ** 2 + (eye_landmarks[1][1] - eye_landmarks[5][1]) ** 2) ** 0.5
    B = ((eye_landmarks[2][0] - eye_landmarks[4][0]) ** 2 + (eye_landmarks[2][1] - eye_landmarks[4][1]) ** 2) ** 0.5
    C = ((eye_landmarks[0][0] - eye_landmarks[3][0]) ** 2 + (eye_landmarks[0][1] - eye_landmarks[3][1]) ** 2) ** 0.5
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3
CLOSED_EYE_DURATION = 1.5
SENSITIVITY = 0.5

blink_count = 0
frame_counter = 0
closed_eye_start_time = None
blink_timestamps = deque()
drowsiness_count = 0
drowsiness_active = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ajuste automático de brillo y contraste
    frame = automatic_brightness_and_contrast(frame)

    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]

        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        for x, y in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if closed_eye_start_time is None:
                closed_eye_start_time = time.time()
            else:
                elapsed_time = time.time() - closed_eye_start_time
                if elapsed_time >= CLOSED_EYE_DURATION * SENSITIVITY:
                    text = "Somnolencia detectada 😴"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    x_pos = width - text_size[0] - 50
                    y_pos = 100
                    cv2.putText(frame, text, (x_pos, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    if not drowsiness_active:
                        drowsiness_count += 1
                        drowsiness_active = True
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
                blink_timestamps.append(time.time())
            frame_counter = 0
            closed_eye_start_time = None
            drowsiness_active = False

        current_time = time.time()
        while blink_timestamps and current_time - blink_timestamps[0] > 60:
            blink_timestamps.popleft()
        blink_frequency = len(blink_timestamps)

        cv2.putText(frame, f"Parpadeos: {blink_count}", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f"Frecuencia: {blink_frequency} por minuto", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f"Somnolencias detectadas: {drowsiness_count}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "No se detecta rostro", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Detección de Fatiga", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
