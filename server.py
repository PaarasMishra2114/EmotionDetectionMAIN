import base64
import os

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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


def find_faces(gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(32, 32))
    if len(faces) > 0:
        return faces
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    faces = face_cascade.detectMultiScale(enhanced, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
    if len(faces) > 0:
        return faces
    enlarged = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    faces = face_cascade.detectMultiScale(enlarged, scaleFactor=1.08, minNeighbors=3, minSize=(32, 32))
    return [(int(x / 1.5), int(y / 1.5), int(width / 1.5), int(height / 1.5)) for x, y, width, height in faces]


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
    faces = find_faces(gray)
    if len(faces) == 0:
        return jsonify(error='No face detected'), 422
    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    face = cv2.resize(gray[y:y + height, x:x + width], (48, 48)).astype('float32') / 255.0
    flipped = cv2.flip(face, 1)
    probabilities = (model(face[np.newaxis, ..., np.newaxis], training=False).numpy()[0] + model(flipped[np.newaxis, ..., np.newaxis], training=False).numpy()[0]) / 2.0
    index = int(np.argmax(probabilities))
    return jsonify(emotion=EMOTIONS[index], confidence=float(probabilities[index]), emotions={name: float(value) for name, value in zip(EMOTIONS, probabilities)}, face={'x': int(x), 'y': int(y), 'width': int(width), 'height': int(height)})


@app.get('/')
def index():
    return send_from_directory(ROOT, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)), debug=False)
