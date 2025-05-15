# 🌾 Crop Disease Detection using Deep Learning

This repository contains a Convolutional Neural Network (CNN) model trained to detect diseases in wheat crops. The model is built and trained using Keras and TensorFlow, and it can be used to classify crop images into healthy or diseased categories.
Made by : Kavay Yadav 2K23CSUN01300
          Pratham 2K23CSUN01302
          Saheb Alam 2K23CSUN01303
---

## 🧠 Model Overview

- **Model Name:** `Cropdiseasedetection_model.h5`
- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input:** Image of wheat crop leaves
- **Output:** Disease category prediction

---

## 📁 Files Included

| File Name                     | Description                                         |
|------------------------------|-----------------------------------------------------|
| `Cropdiseasedetection_model.h5` | Trained Keras CNN model for crop disease detection |
| `app.py` *(optional)*         | Streamlit app file for running the prediction UI   |

---

## 🚀 How to Use the Model

1. Clone the repository

```bash
git clone https://github.com/Kavay2005/ULNN_project.git
cd your-repo-name

2. Install the dependencies

pip install tensorflow numpy pillow streamlit

3. Load the model in your Python code

from tensorflow.keras.models import load_model
model = load_model('Cropdiseasedetection_model.h5')

4. Make predictions on crop images

import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('test_crop.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)
print("Predicted class:", prediction)

