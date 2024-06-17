from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import model_from_json
import base64

app = Flask(__name__)

# Load Face Detection Model
face_cascade_path = r"models\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"Error loading face cascade from {face_cascade_path}")

# Load Anti-Spoofing Model
model_json_path = r'models\antispoofing_model.json'
model_weights_path = r'models\antispoofing_model.h5'

with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights(model_weights_path)
print("Model loaded from disk")

def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x, y, w, h) in faces:
        face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)

        preds = model.predict(resized_face)[0]
        label = 'real' if preds <= 0.5 else 'spoof'
        results.append({
            'label': label,
            'bbox': [int(x), int(y), int(w), int(h)]
        })

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = detect_and_predict(frame)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
