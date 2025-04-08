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

# List of subjects
subjects = ['.NET', 'EVS', 'MOBILE TECHNOLOGY', 'PROJECTLAB', 'ERP']
selected_subject = None  # Store selected subject

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
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

print("Training the recognizer...")
faces, labels, label_dict = prepare_training_data(datasets)
recognizer.train(faces, labels)
print("Training complete!")

# Function to mark attendance
def mark_attendance(name):
    if not selected_subject:
        return f"No subject selected! Attendance cannot be marked."

    attendance_file = f'attendance_{selected_subject.lower()}.csv'

    # Create the attendance file for the subject if it doesn't exist
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Time'])

    # Check if student is already marked in this subject
    with open(attendance_file, 'r', newline='') as file:
        reader = csv.reader(file)
        entries = list(reader)

    if any(row[0] == name for row in entries):
        return f"{name} is already marked for {selected_subject}."

    # Mark attendance
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, current_time])

    return f"Attendance marked for {name} in {selected_subject} at {current_time}."

# Flask route to update selected subject
@app.route('/set_subject', methods=['POST'])
def set_subject():
    global selected_subject
    selected_subject = request.form.get('subject')
    return jsonify({'message': f'Subject set to {selected_subject}'}), 200

# Flask route to stream video
def generate_frames():
    ip_webcam_url = 'http://192.168.189.210:8080/video'
    webcam = cv2.VideoCapture(ip_webcam_url)

    while True:
        success, frame = webcam.read()
        if not success:
            break
        
        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, box_w, box_h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

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
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', subjects=subjects)

if __name__ == '__main__':
    app.run(debug=True)
