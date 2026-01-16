
import os
import subprocess
import sys

# وظيفة لضمان تثبيت المكتبات قبل تشغيل الكود الأساسي
def install_dependencies():
    try:
        import mediapipe
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe", "opencv-python-headless"])

install_dependencies()

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# إعدادات MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

st.set_page_config(page_title="Squat AI Coach", page_icon="🏋️")
st.title("المدرب الذكي لتحليل السكوات 🏋️")

video_file = st.file_uploader("ارفع فيديو التمرين (MP4)...", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    counter = 0 
    stage = None
    min_angle = 180

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                angle = calculate_angle(hip, knee, ankle)
                if angle < min_angle: min_angle = angle
                
                if angle > 160: stage = "up"
                if angle < 90 and stage == 'up':
                    stage = "down"
                    counter += 1
                
                cv2.putText(image, f'Reps: {counter}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except: pass

            st_frame.image(image, channels="BGR")
            
    cap.release()
    st.success(f"التحليل اكتمل! التكرارات: {counter} | أعمق زاوية: {min_angle:.1f}°")
