# import os
# import time
# import urllib.request

# import cv2
# import mediapipe as mp


# MODEL_URL = (
#     "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
#     "hand_landmarker/float16/1/hand_landmarker.task"
# )
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


# def _download_model_if_needed(model_path: str) -> None:
#     if os.path.exists(model_path):
#         return
#     print("Downloading MediaPipe hand model...")
#     urllib.request.urlretrieve(MODEL_URL, model_path)


# def _draw_task_landmarks(img, hand_landmarks):
#     h, w, _ = img.shape
#     connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

#     for lm in hand_landmarks:
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

#     for conn in connections:
#         start = hand_landmarks[conn.start]
#         end = hand_landmarks[conn.end]
#         x1, y1 = int(start.x * w), int(start.y * h)
#         x2, y2 = int(end.x * w), int(end.y * h)
#         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

#     root = hand_landmarks[0]
#     cv2.circle(img, (int(root.x * w), int(root.y * h)), 10, (255, 0, 255), cv2.FILLED)


# cap = cv2.VideoCapture(0)
# use_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

# if use_legacy_solutions:
#     mphands = mp.solutions.hands
#     hands = mphands.Hands(False)
#     mp_draw = mp.solutions.drawing_utils
# else:
#     _download_model_if_needed(MODEL_PATH)
#     base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
#     options = mp.tasks.vision.HandLandmarkerOptions(
#         base_options=base_options,
#         num_hands=2,
#         running_mode=mp.tasks.vision.RunningMode.IMAGE,
#     )
#     hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

# p_time = 0

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     if use_legacy_solutions:
#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_lms in results.multi_hand_landmarks:
#                 for idx, lm in enumerate(hand_lms.landmark):
#                     h, w, _ = img.shape
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     print(idx, cx, cy)
#                     if idx == 0:
#                         cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
#                 mp_draw.draw_landmarks(img, hand_lms, mphands.HAND_CONNECTIONS)
#     else:
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
#         results = hands.detect(mp_image)
#         if results.hand_landmarks:
#             for hand_landmarks in results.hand_landmarks:
#                 for idx, lm in enumerate(hand_landmarks):
#                     h, w, _ = img.shape
#                     print(idx, int(lm.x * w), int(lm.y * h))
#                 _draw_task_landmarks(img, hand_landmarks)

#     c_time = time.time()
#     fps = 1 / (c_time - p_time) if c_time != p_time else 0
#     p_time = c_time

#     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# if not use_legacy_solutions:
#     hands.close()

# cap.release()
# cv2.destroyAllWindows()


import os
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

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


cap = cv2.VideoCapture(0)
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
        num_hands=1,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

prev_x, prev_y = 0, 0
canvas = None
p_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    if canvas is None:
        canvas = np.zeros_like(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if use_legacy_solutions:
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand_lms, mphands.HAND_CONNECTIONS)

            thumb_tip = hand_lms.landmark[4]
            index_tip = hand_lms.landmark[8]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

            cv2.circle(img, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 2)

            if distance < 40:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y

                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 0, 0), 5)
                prev_x, prev_y = index_x, index_y
                cv2.putText(img, "DRAW", (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                prev_x, prev_y = 0, 0
                cv2.putText(img, "MOVE", (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = hands.detect(mp_image)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

            cv2.circle(img, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 2)

            if distance < 40:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y

                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 0, 0), 5)
                prev_x, prev_y = index_x, index_y
                cv2.putText(img, "DRAW", (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                prev_x, prev_y = 0, 0
                cv2.putText(img, "MOVE", (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    img = cv2.addWeighted(img, 1, canvas, 1, 0)

    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(img, "C = clear | Q = quit", (10, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow("Air Draw", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        canvas = np.zeros_like(img)
    elif key == ord("q"):
        break

if not use_legacy_solutions:
    hands.close()

cap.release()
cv2.destroyAllWindows()