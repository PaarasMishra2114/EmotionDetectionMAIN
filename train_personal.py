import argparse
import os
import shutil

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
parser = argparse.ArgumentParser(description='Fine-tune the emotion CNN on one person\'s labeled expressions.')
parser.add_argument('--person', required=True, help='Folder name under data/personal/')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

root = os.path.join('data', 'personal', args.person)
missing = [emotion for emotion in EMOTIONS if not os.path.isdir(os.path.join(root, emotion))]
if missing:
    raise SystemExit(f'Missing expression folders: {", ".join(missing)}')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.25),
    Flatten(), Dense(1024, activation='relu'), Dropout(0.5), Dense(7, activation='softmax')
])
weights = os.path.join('src', 'model.h5')
checkpoint = os.path.join('src', 'personal_best.weights.h5')
starting_weights = checkpoint if os.path.exists(checkpoint) else weights
if os.path.exists(starting_weights):
    model.load_weights(starting_weights)

images = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, rotation_range=8, zoom_range=0.08, horizontal_flip=True)
train = images.flow_from_directory(root, target_size=(48, 48), batch_size=args.batch_size, color_mode='grayscale', class_mode='categorical', classes=EMOTIONS, subset='training')
validation = images.flow_from_directory(root, target_size=(48, 48), batch_size=args.batch_size, color_mode='grayscale', class_mode='categorical', classes=EMOTIONS, subset='validation')
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00005), metrics=['accuracy'])
model.fit(train, epochs=args.epochs, validation_data=validation, callbacks=[ModelCheckpoint(checkpoint, save_best_only=True, save_weights_only=True, monitor='val_accuracy')])
model.load_weights(checkpoint)
print(f'Best model saved to {checkpoint}')
