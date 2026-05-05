import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Face Mesh (468 個特徵點，能更精準地抓取臉部角度)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 打開攝像頭
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 轉換 BGR 到 RGB 以供 MediaPipe 使用
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
                # 定義一個不顯示特徵點（只顯示線）的規格
                # circle_radius=0 可以隱藏原本煩人的紅點
                no_dots_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0)

                # 1. 繪製全臉網格 (淺灰色，極細)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=no_dots_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1))

                # 2. 用螢光綠 (Neon Green) 繪製眼睛輪廓
                # 分別繪製左眼、右眼以及眼眶周邊
                eye_connections = [mp_face_mesh.FACEMESH_LEFT_EYE, 
                                   mp_face_mesh.FACEMESH_RIGHT_EYE,
                                   mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                                   mp_face_mesh.FACEMESH_RIGHT_EYEBROW]
                
                for connection in eye_connections:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=connection,
                        landmark_drawing_spec=no_dots_spec,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5))

                # 3. 用螢光紅 (Neon Red) 繪製嘴巴輪廓
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=no_dots_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6))

    cv2.imshow('Face Skeleton & Feature Tracking', frame)

    if cv2.waitKey(5) & 0xFF == 27: # 按下 ESC 退出
        break

cap.release()
cv2.destroyAllWindows()
