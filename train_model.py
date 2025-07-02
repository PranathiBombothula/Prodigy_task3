import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

IMG_SIZE = 64
DATASET_PATH = "C:\\Users\\kiran\\Downloads\\cats_dogs_svm_project\\cats_dogs_svm\\dataset\\train"

X = []
y = []

print("Loading images...")
for img_name in os.listdir(DATASET_PATH):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    label = 0 if "cat" in img_name.lower() else 1
    img_path = os.path.join(DATASET_PATH, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping corrupted image: {img_name}")
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X.append(img.flatten())
    y.append(label)


X = np.array(X)
y = np.array(y)

print("Training SVM model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import os
os.makedirs("model", exist_ok=True)
joblib.dump(clf, os.path.join("model", "svm_model.pkl"))


