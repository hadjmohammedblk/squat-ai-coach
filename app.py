import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile

st.set_page_config(page_title="AI Squat Coach")
st.title("المدرب الذكي لتحليل السكوات 🏋️")

# استدعاء الأدوات بطريقة متوافقة مع النسخ القديمة والجديدة
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    st.error("خطأ في تحميل مكتبة Mediapipe. يرجى التأكد من نسخة Python.")

video_file = st.file_uploader("ارفع فيديو التمرين (MP4)", type=['mp4'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            st_frame.image(frame, channels="BGR")
    cap.release()            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    # تحديد نقاط المفصل
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    
                    angle = calculate_angle(hip, knee, ankle)
                    if angle > 160: stage = "up"
                    if angle < 90 and stage == 'up':
                        stage, counter = "down", counter + 1
                    
                    cv2.putText(image, f'Reps: {counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except Exception: pass
            
            st_frame.image(image, channels="BGR")
    cap.release()            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    hip = [landmarks[BasePose.PoseLandmark.LEFT_HIP].x, landmarks[BasePose.PoseLandmark.LEFT_HIP].y]
                    knee = [landmarks[BasePose.PoseLandmark.LEFT_KNEE].x, landmarks[BasePose.PoseLandmark.LEFT_KNEE].y]
                    ankle = [landmarks[BasePose.PoseLandmark.LEFT_ANKLE].x, landmarks[BasePose.PoseLandmark.LEFT_ANKLE].y]
                    
                    angle = calculate_angle(hip, knee, ankle)
                    if angle > 160: stage = "up"
                    if angle < 90 and stage == 'up':
                        stage, counter = "down", counter + 1
                    
                    cv2.putText(image, f'Reps: {counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    BaseDrawing.draw_landmarks(image, results.pose_landmarks, BasePose.POSE_CONNECTIONS)
                except Exception: pass
            
            st_frame.image(image, channels="BGR")
    cap.release()    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    # إحداثيات الورك والركبة والكاحل
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    
                    angle = calculate_angle(hip, knee, ankle)
                    min_angle = min(min_angle, angle)
                    
                    if angle > 160: stage = "up"
                    if angle < 90 and stage == 'up':
                        stage = "down"
                        counter += 1
                    
                    # رسم البيانات على الصورة
                    cv2.putText(image, f'Reps: {counter}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except Exception:
                    pass

            st_frame.image(image, channels="BGR")
            
    cap.release()
    st.success(f"اكتمل التحليل! التكرارات: {counter} | أعمق زاوية وصلتها: {min_angle:.1f}°")
