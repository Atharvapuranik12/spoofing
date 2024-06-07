from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
app = Flask(__name__)

# Load Face Detection Model
face_cascade_path = r"C:\Users\ss\OneDrive\Desktop\spoof\models\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"Error loading face cascade from {face_cascade_path}")

# Load Anti-Spoofing Model
model_json_path = r'C:\Users\ss\OneDrive\Desktop\spoof\antispoofing_model.json'
model_weights_path = r'C:\Users\ss\OneDrive\Desktop\spoof\antispoofing_model.h5'

with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights(model_weights_path)
print("Model loaded from disk")

def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)

        preds = model.predict(resized_face)[0]
        print(f"Predictions: {preds}")  # Debug: Print predictions

        label = 'real' if preds <= 0.5 else 'spoof'
        color = (0, 255, 0) if preds <= 0.5 else (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
