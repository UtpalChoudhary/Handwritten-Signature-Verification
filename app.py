import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

genuine_dir = 'static/Dataset/'

def resize_image(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height))


def check_similarity(image1, image2):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    similarity_score = ssim(gray_image1, gray_image2)
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify_signature', methods=['POST'])
def verify_signature():
    if 'file' not in request.files:
        return jsonify({'result': 'No file uploaded'})
    
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'result': 'No file selected'})
    

    uploaded_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    sample_image = cv2.imread(os.path.join(genuine_dir, os.listdir(genuine_dir)[0]))
    target_width, target_height = sample_image.shape[1], sample_image.shape[0]
    
    uploaded_image_resized = resize_image(uploaded_image, target_width, target_height)
    
    for filename in os.listdir(genuine_dir):
        genuine_image = cv2.imread(os.path.join(genuine_dir, filename))
        similarity_score = check_similarity(uploaded_image_resized, genuine_image)
        if similarity_score >= 0.7:
            return jsonify({'result': 'Verified'})
    
    return jsonify({'result': 'Not Verified'})

if __name__ == '__main__':
    app.run(debug=True)

