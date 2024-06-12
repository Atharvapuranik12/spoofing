from flask import Flask, render_template, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load Face Detection Model
face_cascade_path = os.path.join('models', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"Error: Failed to load face cascade from {face_cascade_path}")
    exit(1)

def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Perform basic anti-spoofing heuristic
        if len(faces) > 1:  # Check for multiple faces
            label = 'poof'
            color = (0, 0, 255)
        else:  # Check for eye blink detection
            # Code for eye blink detection goes here
            label = 'eal'
            color = (0, 255, 0)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def generate_frames():
    # Use the default camera device (typically 0)
    video_device = 0
    print(f"Trying to open video device: {video_device}")
    video = cv2.VideoCapture(video_device)
    if not video.isOpened():
        print(f"Error: Could not open video source at {video_device}")
        return

    print("Video source opened successfully!")
    while True:
        success, frame = video.read()
        if not success:
            print("Error: Failed to read frame from video source.")
            break

        frame = detect_and_predict(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
