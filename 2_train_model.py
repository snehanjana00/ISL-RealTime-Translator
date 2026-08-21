import os
import pickle
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'
data = []
labels = []

print("Extracting features from captured images...")

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Resize image to 64x64 for feature extraction
        img_resized = cv2.resize(img, (64, 64))
        data.append(img_resized.flatten())
        labels.append(dir_)

data = np.asarray(data)
labels = np.asarray(labels)

print("Training Machine Learning model...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"\n=================================")
print(f"Model Accuracy: {score * 100:.2f}%")
print(f"=================================\n")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("SUCCESS: model.p saved successfully!")