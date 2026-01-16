import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

# إعداد الصفحة
st.set_page_config(page_title="AI Squat Coach", page_icon="🏋️")
st.title("المدرب الذكي لتحليل السكوات 🏋️")

# تعريف أدوات MediaPipe بطريقة متوافقة مع السيرفر
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

video_file = st.file_uploader("ارفع فيديو التمرين (MP4)...", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    counter, stage, min_angle = 0, None, 180

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    
                    angle = calculate_angle(hip, knee, ankle)
                    min_angle = min(min_angle, angle)
                    if angle > 160: stage = "up"
                    if angle < 90 and stage == 'up':
                        stage, counter = "down", counter + 1
                    
                    cv2.putText(image, f'Reps: {counter}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except: pass
            st_frame.image(image, channels="BGR")
    cap.release()
    st.success(f"التحليل اكتمل! التكرارات: {counter} | أعمق زاوية: {min_angle:.1f}°")            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # نقاط الجسم (الورك، الركبة، الكاحل)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                angle = calculate_angle(hip, knee, ankle)
                min_angle = min(min_angle, angle)             
                if angle > 160: stage = "up"
                if angle < 90 and stage == 'up':
                    stage, counter = "down", counter + 1   
                cv2.putText(image, f'Reps: {counter}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st_frame.image(image, channels="BGR")
    cap.release()
    st.success(f"التحليل اكتمل! التكرارات: {counter} | أعمق زاوية: {min_angle:.1f}°")
