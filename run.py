from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

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
            label = 'spoof'
            color = (0, 0, 255)
        else:  # Check for eye blink detection
            # Code for eye blink detection goes here
            label = 'real'
            color = (0, 255, 0)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def generate_frames():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video source.")
        return

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
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'frame' not in request.files:
        return "No frame found", 400

    frame = request.files['frame'].read()
    npimg = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    processed_frame = detect_and_predict(img)
    _, buffer = cv2.imencode('.jpg', processed_frame)
    io_buf = BytesIO(buffer)
    return send_file(io_buf, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
