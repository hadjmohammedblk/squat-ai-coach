import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# إعداد ميديا بايب
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180 else angle

st.set_page_config(page_title="AI Fitness Coach", layout="wide")
st.title("🏋️‍♂️ المدرب الذكي لتحليل السكوات")

uploaded_file = st.file_uploader("ارفع فيديو تمرين السكوات (MP4)", type=["mp4", "mov"])

if uploaded_file:
    # حفظ الفيديو المرفوع مؤقتاً
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    all_angles = []
    
    with st.spinner('جاري تحليل الأداء...'):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # نقاط الجانب الأيسر (الورك 23، الركبة 25، الكاحل 27)
                hip = [lm[23].x, lm[23].y]
                knee = [lm[25].x, lm[25].y]
                ankle = [lm[27].x, lm[27].y]
                all_angles.append(calculate_angle(hip, knee, ankle))
        cap.release()

    if all_angles:
        # حساب التكرارات والزاوية الدنيا
        reps = 0
        stage = "up"
        for angle in all_angles:
            if angle < 90: stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"
                reps += 1
        
        best_angle = min(all_angles)
        
        # عرض النتائج في 
