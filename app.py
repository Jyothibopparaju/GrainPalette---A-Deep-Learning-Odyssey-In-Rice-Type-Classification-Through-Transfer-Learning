from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import traceback

app = Flask(__name__)

# Load your trained model (make sure rice.h5 is in your project directory)
model = load_model('rice.h5')

# Update class names based on your model's actual classes
class_names = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'Red']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            # Ensure static directory exists
            if not os.path.exists('static'):
                os.makedirs('static')

            # Create unique filename and save
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            img_path = os.path.join('static', filename)
            file.save(img_path)

            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))  # Adjust size as per your model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction[0])]

            # Render result page with prediction and image
            return render_template('results.html',
                                   prediction=predicted_class,
                                   image_url=url_for('static', filename=filename))

        return render_template('index.html', error="Something went wrong")

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
