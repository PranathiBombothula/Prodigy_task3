import cv2
import numpy as np
import joblib

IMG_SIZE = 64
model = joblib.load("C:\\Users\\kiran\\Downloads\\cats_dogs_svm_project\\cats_dogs_svm\\model\\svm_model.pkl")

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1)
    pred = model.predict(img)[0]
    return "Dog" if pred == 1 else "Cat"
