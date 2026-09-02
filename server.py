import base64
import os

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
app = Flask(__name__, static_folder=ROOT, static_url_path='')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax'),
])
weights_path = os.path.join(SRC, 'personal_best.weights.h5')
if not os.path.exists(weights_path):
    weights_path = os.path.join(SRC, 'model.h5')
model.load_weights(weights_path)
face_cascade = cv2.CascadeClassifier(os.path.join(SRC, 'haarcascade_frontalface_default.xml'))


def decode_image(data_url):
    try:
        encoded = data_url.split(',', 1)[-1]
        return cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, TypeError):
        return None


@app.post('/api/predict')
def predict():
    payload = request.get_json(silent=True) or {}
    image = decode_image(payload.get('image', ''))
    if image is None:
        return jsonify(error='Invalid image'), 400
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        enhanced = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(enhanced, scaleFactor=1.08, minNeighbors=3, minSize=(32, 32))
    if len(faces) == 0:
        return jsonify(error='No face detected'), 422
    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    face = cv2.resize(gray[y:y + height, x:x + width], (48, 48)).astype('float32') / 255.0
    probabilities = model.predict(face[np.newaxis, ..., np.newaxis], verbose=0)[0]
    index = int(np.argmax(probabilities))
    return jsonify(emotion=EMOTIONS[index], confidence=float(probabilities[index]), emotions={name: float(value) for name, value in zip(EMOTIONS, probabilities)}, face={'x': int(x), 'y': int(y), 'width': int(width), 'height': int(height)})


@app.get('/')
def index():
    return send_from_directory(ROOT, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)
