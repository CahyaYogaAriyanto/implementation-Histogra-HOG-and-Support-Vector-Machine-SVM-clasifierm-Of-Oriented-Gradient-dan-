from flask import Flask, request, render_template, send_from_directory,jsonify,flash,redirect,url_for
import os
import joblib
from skimage.feature import hog
import cv2
import numpy as np
from skimage import color, io, transform

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

svm_model = joblib.load('svm_model.pkl')
def extract_hog_features_from_file(image_path):
    image = io.imread(image_path)
    # Hapus saluran alpha jika ada
    if image.shape[-1] == 4:
        image = image[:, :, :-1] 
    gray_image = color.rgb2gray(image)  
    resized_image = transform.resize(gray_image, (128, 128))  

    features = hog(resized_image, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    return features

@app.route('/')
def index():
    return render_template('uploaded.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return render_template('uploaded.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        
        return render_template('uploaded.html', alert_message='No selected file.')
    if file:
        # Proses gambar dan prediksi
        image_path = os.path.join('static/uploads', file.filename)
        file.save(image_path)

        # Extract HOG features 
        hog_features = extract_hog_features_from_file(image_path)

        # Konversi dari ndarray ke list dan ubah elemen ke float
        hog_features_list = hog_features.tolist()  
        hog_features_list = [float(value) for value in hog_features_list]  
        
        # Prediksi
        prediction = svm_model.predict([hog_features])
  
        return render_template('predict.html', prediction=prediction,image_url=image_path)
    flash('Invalid request. Please try again.', 'error')
    return jsonify({'error': 'Invalid request'}), 400


if __name__ == '__main__':
    app.run(debug=True)
