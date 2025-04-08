from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
import csv
from datetime import datetime
import mediapipe as mp

app = Flask(__name__)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Dataset folder
datasets = 'Dataset'
subjects = ['.NET', 'EVS', 'MOBILE TECHNOLOGY', 'PROJECTLAB', 'ERP']

# Global variables
selected_subject = None
video_active = False  # Controls whether the video feed starts

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def prepare_training_data(datasets):
    faces, labels = [], []
    label_dict = {}
    label_id = 0
    for subdir in os.listdir(datasets):
        if os.path.isdir(os.path.join(datasets, subdir)):
            label_dict[label_id] = subdir
            for filename in os.listdir(os.path.join(datasets, subdir)):
                filepath = os.path.join(datasets, subdir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_id)
            label_id += 1
    return np.array(faces), np.array(labels), label_dict

faces, labels, label_dict = prepare_training_data(datasets)
recognizer.train(faces, labels)

def mark_attendance(name):
    if not selected_subject:
        return "No subject selected!"
    attendance_file = f'attendance_{selected_subject.lower()}.csv'
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Time'])
    with open(attendance_file, 'r', newline='') as file:
        reader = csv.reader(file)
        entries = list(reader)
    if any(row[0] == name for row in entries):
        return f"{name} is already marked for {selected_subject}."
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, current_time])
    return f"Attendance marked for {name} in {selected_subject}."

def generate_frames():
    global video_active
    ip_webcam_url = 'http://192.168.189.210:8080/video'
    webcam = None

    while True:
        if not video_active:  # Only start webcam if subject is selected
            continue
        
        if webcam is None:
            webcam = cv2.VideoCapture(ip_webcam_url)

        success, frame = webcam.read()
        if not success:
            break

        # Resize the frame to a rectangular shape (4:3 aspect ratio)
        frame = cv2.resize(frame, (640, 480))  # Change to (500, 500) for square

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, box_w, box_h = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))
                x, y = max(0, x), max(0, y)
                box_w, box_h = min(w - x, box_w), min(h - y, box_h)
                
                face = frame[y:y+box_h, x:x+box_w]
                if face.size == 0:
                    continue
                
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (130, 100))
                
                label, confidence = recognizer.predict(face_resized)
                name = label_dict.get(label, "Unknown")

                if confidence < 100:
                    mark_attendance(name)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', subjects=subjects)

@app.route('/video_feed')
def video_feed():
    if not video_active:  # Don't start if no subject is selected
        return jsonify({"error": "Select a subject first!"}), 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_subject', methods=['POST'])
def select_subject():
    global selected_subject, video_active
    selected_subject = request.form['subject']
    video_active = True  # Enable webcam feed
    return jsonify({'message': f'Subject set to {selected_subject}. You can start attendance.'})

if __name__ == '__main__':
    app.run(debug=True)
