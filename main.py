from flask import Flask, render_template, request
from googletrans import Translator
from tensorflow.keras.models import load_model
# from keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
translator = Translator()

# Load your trained traffic sign recognition model
model = load_model(r'CC:\Users\mrudu\OneDrive\Documents\project\Final\traffic-sign.h5')

# Define your labels
labels = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing veh over 3.5 tons", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5 tons prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End no passing veh > 3.5 tons"
]

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image and language code
    file = request.files['image']
    lang = request.form['lang']
    
    # Load the image
    img = Image.open(file.stream)
    img = img.resize((30, 30))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    pred = np.argmax(model.predict(img_array), axis=-1)[0]
    predicted_label = labels[pred]

    # Translate the predicted label
    translated = translator.translate(predicted_label, dest=lang)

    return render_template('index1.html', original=predicted_label, translated=translated.text, lang=lang)

if __name__ == '__main__':
    app.run(debug=True)
