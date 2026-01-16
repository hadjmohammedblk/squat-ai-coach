import os
# إجبار السيرفر على تثبيت المكتبات اللازمة عند التشغيل
os.system("pip install mediapipe opencv-python-headless")

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile

# إعدادات مكتبة MediaPipe لتتبع الجسم
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

st.set_page_config(page_title="AI Fitness Coach", page_icon="🏋️")
st.title("المدرب الذكي لتحليل تمرين القرفصاء (Squat) 🏋️")
st.write("ارفع فيديو التمرين الخاص بك ليقوم الذكاء الاصطناعي بتحليل الأداء وعدّ التكرارات.")

video_file = st.file_uploader("ارفع فيديو (MP4)...", type=['mp4', 'mov', 'avi'])

if video_file is not None:
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
            if not ret:
                break
            
            # معالجة الصورة
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                # إحداثيات الورك، الركبة، والكاحل
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                angle = calculate_angle(hip, knee, ankle)
                if angle < min_angle:
                    min_angle = angle
                
                # منطق عد التكرارات
                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == 'up':
                    stage = "down"
                    counter += 1
                
                # رسم البيانات على الفيديو
                cv2.putText(image, f'Reps: {counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Angle: {int(angle)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except:
                pass

            st_frame.image(image, channels="BGR")
            
    cap.release()
    st.success(f"تم التحليل بنجاح! إجمالي العدات: {counter}")
    st.info(f"أعمق زاوية نزول تم تسجيلها: {min_angle:.1f} درجة")
