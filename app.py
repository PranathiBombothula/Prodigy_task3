from flask import Flask, render_template, request
import os
from utils import predict_image
from werkzeug.utils import secure_filename

import os

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = 'C:\\Users\\kiran\\Downloads\\cats_dogs_svm_project\\cats_dogs_svm\\static\\uploads'

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("index.html", prediction=prediction, image_path=filepath)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
