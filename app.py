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


def _distance_pixels(point_a, point_b, frame_width, frame_height):
    x1, y1 = int(point_a.x * frame_width), int(point_a.y * frame_height)
    x2, y2 = int(point_b.x * frame_width), int(point_b.y * frame_height)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, (x1, y1), (x2, y2)


cap = cv2.VideoCapture(0)

# camera size better kar do
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

use_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

if use_legacy_solutions:
    mphands = mp.solutions.hands
    hands = mphands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
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

window_name = "Image"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

p_time = 0
prev_x, prev_y = 0, 0
canvas = None
fullscreen = False

draw_color = (255, 0, 0)
brush_thickness = 6
eraser_radius = 35

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_hands = 0
    mode_text = "MOVE"
    drawing_active = False

    if use_legacy_solutions:
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            detected_hands = len(results.multi_hand_landmarks)

            for hand_index, hand_lms in enumerate(results.multi_hand_landmarks):
                # complete hand map
                mp_draw.draw_landmarks(img, hand_lms, mphands.HAND_CONNECTIONS)

                # sirf first hand se draw/erase
                if hand_index == 0:
                    thumb_tip = hand_lms.landmark[4]
                    index_tip = hand_lms.landmark[8]
                    middle_tip = hand_lms.landmark[12]

                    thumb_index_dist, thumb_xy, index_xy = _distance_pixels(thumb_tip, index_tip, w, h)
                    index_middle_dist, _, middle_xy = _distance_pixels(index_tip, middle_tip, w, h)

                    ix, iy = index_xy
                    mx, my = middle_xy

                    cv2.circle(img, (ix, iy), 10, (0, 255, 255), cv2.FILLED)
                    cv2.circle(img, (mx, my), 10, (0, 200, 255), cv2.FILLED)

                    # erase mode: index + middle close
                    if index_middle_dist < 35:
                        mode_text = "ERASE"
                        cv2.circle(img, (ix, iy), eraser_radius, (0, 0, 255), 2)
                        cv2.circle(canvas, (ix, iy), eraser_radius, (0, 0, 0), -1)
                        prev_x, prev_y = 0, 0

                    # draw mode: thumb + index close
                    elif thumb_index_dist < 40:
                        mode_text = "DRAW"
                        drawing_active = True
                        cv2.line(img, thumb_xy, index_xy, (0, 255, 255), 2)

                        # smooth drawing
                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = ix, iy

                        smooth_x = int((prev_x + ix) / 2)
                        smooth_y = int((prev_y + iy) / 2)

                        cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, brush_thickness)
                        prev_x, prev_y = smooth_x, smooth_y

                    else:
                        mode_text = "MOVE"
                        prev_x, prev_y = 0, 0

    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = hands.detect(mp_image)

        if results.hand_landmarks:
            detected_hands = len(results.hand_landmarks)

            for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
                _draw_task_landmarks(img, hand_landmarks)

                if hand_index == 0:
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    middle_tip = hand_landmarks[12]

                    thumb_index_dist, thumb_xy, index_xy = _distance_pixels(thumb_tip, index_tip, w, h)
                    index_middle_dist, _, middle_xy = _distance_pixels(index_tip, middle_tip, w, h)

                    ix, iy = index_xy
                    mx, my = middle_xy

                    cv2.circle(img, (ix, iy), 10, (0, 255, 255), cv2.FILLED)
                    cv2.circle(img, (mx, my), 10, (0, 200, 255), cv2.FILLED)

                    if index_middle_dist < 35:
                        mode_text = "ERASE"
                        cv2.circle(img, (ix, iy), eraser_radius, (0, 0, 255), 2)
                        cv2.circle(canvas, (ix, iy), eraser_radius, (0, 0, 0), -1)
                        prev_x, prev_y = 0, 0

                    elif thumb_index_dist < 40:
                        mode_text = "DRAW"
                        drawing_active = True
                        cv2.line(img, thumb_xy, index_xy, (0, 255, 255), 2)

                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = ix, iy

                        smooth_x = int((prev_x + ix) / 2)
                        smooth_y = int((prev_y + iy) / 2)

                        cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, brush_thickness)
                        prev_x, prev_y = smooth_x, smooth_y

                    else:
                        mode_text = "MOVE"
                        prev_x, prev_y = 0, 0

    # drawing ko clearly show karne ke liye
    overlay = cv2.addWeighted(img, 1.0, canvas, 1.0, 0)

    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time

    cv2.putText(overlay, f"Hands: {detected_hands}", (15, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(overlay, f"FPS: {int(fps)}", (15, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(overlay, f"Mode: {mode_text}", (15, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.putText(overlay, "Draw: Thumb+Index", (15, 145), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(overlay, "Erase: Index+Middle", (15, 180), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(overlay, "C=clear | F=fullscreen | Q=quit", (15, 215), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # preview ko bada karke show karo
    display_img = cv2.resize(overlay, (1280, 720))
    cv2.imshow(window_name, display_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    elif key == ord("f"):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)

    elif key == ord("q"):
        break

if not use_legacy_solutions:
    hands.close()

cap.release()
cv2.destroyAllWindows()