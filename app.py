import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow import keras
from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "crop_disease_model.keras")
CLASS_PATH = os.path.join(BASE, "class_names.json")

# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# --------------------------------------------------
# LOAD AI MODEL
# --------------------------------------------------

print("Loading AI model...")

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("AI model loaded successfully.")
print("Classes:", class_names)

# --------------------------------------------------
# FARMER ADVISORY
# --------------------------------------------------

ADVISORY = {

    "Potato___Early_blight": {
        "disease": "Potato Early Blight",
        "advice": [
            "Remove severely affected leaves and plant debris.",
            "Avoid unnecessary wetting of potato leaves.",
            "Maintain proper spacing between plants for good air circulation.",
            "Monitor nearby plants for similar symptoms."
        ]
    },

    "Potato___Late_blight": {
        "disease": "Potato Late Blight",
        "advice": [
            "Remove severely infected plant material where practical.",
            "Avoid prolonged leaf wetness and improve field ventilation.",
            "Monitor the crop regularly for rapid disease spread.",
            "Consult a local agricultural expert for suitable disease-management measures."
        ]
    },

    "Potato___healthy": {
        "disease": "Healthy Potato Leaf",
        "advice": [
            "The leaf appears healthy according to the AI prediction.",
            "Continue regular crop monitoring.",
            "Maintain proper irrigation, nutrition and field hygiene.",
            "Check new leaves regularly for changes."
        ]
    },

    "Tomato___Bacterial_spot": {
        "disease": "Tomato Bacterial Spot",
        "advice": [
            "Remove severely affected leaves where practical.",
            "Avoid unnecessary overhead watering.",
            "Keep foliage as dry as possible.",
            "Monitor nearby plants for similar symptoms."
        ]
    },

    "Tomato___Early_blight": {
        "disease": "Tomato Early Blight",
        "advice": [
            "Remove severely affected leaves and plant debris.",
            "Maintain adequate spacing and air circulation.",
            "Avoid unnecessary wetting of foliage.",
            "Monitor lower leaves and nearby plants regularly."
        ]
    },

    "Tomato___Late_blight": {
        "disease": "Tomato Late Blight",
        "advice": [
            "Inspect nearby plants because the disease can spread quickly.",
            "Remove severely affected plant material where practical.",
            "Avoid prolonged leaf wetness and improve air circulation.",
            "Consult a local agricultural expert for appropriate disease-management measures."
        ]
    },

    "Tomato___Leaf_Mold": {
        "disease": "Tomato Leaf Mold",
        "advice": [
            "Improve ventilation around tomato plants.",
            "Avoid unnecessary overhead watering.",
            "Remove severely affected leaves where practical.",
            "Monitor nearby plants regularly."
        ]
    },

    "Tomato___healthy": {
        "disease": "Healthy Tomato Leaf",
        "advice": [
            "The leaf appears healthy according to the AI prediction.",
            "Continue regular crop monitoring.",
            "Maintain proper irrigation and plant nutrition.",
            "Check new leaves regularly for symptoms."
        ]
    }
}

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "leaf_image" not in request.files:
        return render_template(
            "index.html",
            error="Please select a leaf image."
        )

    file = request.files["leaf_image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select a leaf image."
        )

    try:
        # Read uploaded image
        image = Image.open(file.stream).convert("RGB")

        # Resize exactly as used during model development
        image = image.resize((224, 224))

        # IMPORTANT:
        # The trained model already contains MobileNetV2
        # preprocess_input internally.
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # AI prediction
        prediction = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index] * 100)

        # Advisory
        information = ADVISORY.get(
            predicted_class,
            {
                "disease": predicted_class,
                "advice": [
                    "Monitor the crop regularly.",
                    "Consult a local agricultural expert if symptoms continue."
                ]
            }
        )

        # Crop name
        if predicted_class.startswith("Potato"):
            crop = "Potato"
        else:
            crop = "Tomato"

        return render_template(
            "index.html",
            result=True,
            crop=crop,
            disease=information["disease"],
            confidence=round(confidence, 2),
            advice=information["advice"]
        )

    except Exception as e:

        print("Prediction error:", e)

        return render_template(
            "index.html",
            error="Unable to analyze this image. Please try another clear leaf image."
        )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )