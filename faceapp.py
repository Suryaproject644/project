from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import mediapipe as mp

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# IP Webcam URL
IP_WEBCAM_URL = 'http://192.168.189.210:8080/video'

@app.route('/')
def index():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    if not name:
        return "Name is required", 400

    datasets = 'Dataset'
    path = os.path.join(datasets, name)
    os.makedirs(path, exist_ok=True)

    webcam = cv2.VideoCapture(IP_WEBCAM_URL)
    if not webcam.isOpened():
        return "Could not open webcam. Check URL.", 500

    count = 1
    (width, height) = (130, 100)

    while count <= 100:
        success, frame = webcam.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                x, y = max(0, x), max(0, y)
                box_w = min(w - x, box_w)
                box_h = min(h - y, box_h)

                face = frame[y:y+box_h, x:x+box_w]
                if face.size == 0:
                    continue

                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (width, height))
                cv2.imwrite(f"{path}/{count}.png", face_resized)
                count += 1

                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)
                cv2.imshow('Capturing Faces', frame)

        if cv2.waitKey(10) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

    return f"Dataset created for {name} with 50 images."

if __name__ == '__main__':
    app.run(debug=True)
