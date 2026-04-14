import os
import time
import urllib.request

import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _download_model_if_needed(model_path: str) -> None:
    if os.path.exists(model_path):
        return
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MODEL_URL, model_path)


def _draw_task_landmarks(img, hand_landmarks):
    h, w, _ = img.shape
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

    for conn in connections:
        start = hand_landmarks[conn.start]
        end = hand_landmarks[conn.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    root = hand_landmarks[0]
    cv2.circle(img, (int(root.x * w), int(root.y * h)), 10, (255, 0, 255), cv2.FILLED)


# Change the video file path to your own file
video_path = "a.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details for the output video
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video_path = "output_video.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

use_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

if use_legacy_solutions:
    mphands = mp.solutions.hands
    hands = mphands.Hands(False)
    mp_draw = mp.solutions.drawing_utils
else:
    _download_model_if_needed(MODEL_PATH)
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

p_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if use_legacy_solutions:
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                for idx, lm in enumerate(hand_lms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(idx, cx, cy)
                    if idx == 0:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                mp_draw.draw_landmarks(img, hand_lms, mphands.HAND_CONNECTIONS)
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = hands.detect(mp_image)
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for idx, lm in enumerate(hand_landmarks):
                    h, w, _ = img.shape
                    print(idx, int(lm.x * w), int(lm.y * h))
                _draw_task_landmarks(img, hand_landmarks)

    c_time = time.time()
    display_fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time

    cv2.putText(img, str(int(display_fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if not use_legacy_solutions:
    hands.close()

cap.release()
out.release()
cv2.destroyAllWindows()