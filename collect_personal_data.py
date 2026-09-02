import argparse
import os
import time

import cv2

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
parser = argparse.ArgumentParser(description='Collect labeled webcam photos for personal emotion training.')
parser.add_argument('--person', default='me', help='Person folder name')
parser.add_argument('--per-expression', type=int, default=110, help='Photos to collect for each expression')
parser.add_argument('--camera', type=int, default=0, help='Camera device index')
args = parser.parse_args()

safe_person = ''.join(character for character in args.person if character.isalnum() or character in ('-', '_')) or 'me'
root = os.path.join('data', 'personal', safe_person)
cascade = cv2.CascadeClassifier(os.path.join('src', 'haarcascade_frontalface_default.xml'))
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit('Could not open the camera. Check camera permissions or try --camera 1.')

try:
    for emotion in EMOTIONS:
        folder = os.path.join(root, emotion)
        os.makedirs(folder, exist_ok=True)
        existing = len([name for name in os.listdir(folder) if name.lower().endswith('.jpg')])
        count = existing
        print(f'\nExpression: {emotion} ({count}/{args.per_expression})')
        print('Press SPACE to save a frame, N to skip to the next expression, or Q to quit.')
        while count < args.per_expression:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
            for x, y, width, height in faces[:1]:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 210, 170), 2)
            cv2.putText(frame, f'{emotion}: {count}/{args.per_expression}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 210, 170), 2)
            cv2.imshow('Personal emotion data collector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt
            if key == ord('n'):
                break
            if key == 32:
                if len(faces) == 0:
                    print('No face detected; move closer or improve lighting.')
                    continue
                x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
                face = frame[y:y + height, x:x + width]
                filename = os.path.join(folder, f'{count + 1:04d}.jpg')
                cv2.imwrite(filename, face)
                count += 1
                print(f'Saved {emotion}: {count}/{args.per_expression}')
                time.sleep(.15)
finally:
    cap.release()
    cv2.destroyAllWindows()
print(f'\nFinished. Labeled data saved under data/personal/{safe_person}/')
