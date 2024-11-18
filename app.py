import flask 
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = YOLO('/content/yolov5best.pt')  # Replace with your model path

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Save and preprocess the image
    file_path = f"temp/{file.filename}"
    os.makedirs('temp', exist_ok=True)
    file.save(file_path)

    # Run inference
    results = model.predict(source=file_path, save=False, conf=0.5)

    # Parse results
    predictions = results[0].boxes.xywh.tolist()  # Example: bounding boxes
    os.remove(file_path)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
