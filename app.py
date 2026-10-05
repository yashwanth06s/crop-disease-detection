import os
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from ai_edge_litert.interpreter import Interpreter

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE, "crop_disease_model.tflite")
CLASS_PATH = os.path.join(BASE, "class_names.json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

print("Loading LiteRT model...")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("LiteRT model loaded successfully.")

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Classes:", class_names)


ADVICE = {

    "Potato___Early_blight": [
        "Remove severely affected leaves and plant debris.",
        "Avoid unnecessary wetting of potato leaves.",
        "Maintain proper spacing between plants for good air circulation.",
        "Monitor nearby plants for similar symptoms."
    ],

    "Potato___Late_blight": [
        "Remove severely infected plant material where practical.",
        "Avoid prolonged leaf wetness and improve field ventilation.",
        "Monitor the crop regularly for rapid disease spread.",
        "Consult a local agricultural expert for suitable disease-management measures."
    ],

    "Potato___healthy": [
        "The leaf appears healthy according to the AI prediction.",
        "Continue regular crop monitoring.",
        "Maintain proper irrigation, nutrition and field hygiene.",
        "Check new leaves regularly for changes."
    ],

    "Tomato___Bacterial_spot": [
        "Remove severely affected leaves where practical.",
        "Avoid unnecessary overhead watering.",
        "Keep foliage as dry as possible.",
        "Monitor nearby plants for similar symptoms."
    ],

    "Tomato___Early_blight": [
        "Remove severely affected leaves and plant debris.",
        "Maintain adequate spacing and air circulation.",
        "Avoid unnecessary wetting of foliage.",
        "Monitor lower leaves and nearby plants regularly."
    ],

    "Tomato___Late_blight": [
        "Inspect nearby plants because the disease can spread quickly.",
        "Remove severely affected plant material where practical.",
        "Avoid prolonged leaf wetness and improve air circulation.",
        "Consult a local agricultural expert for appropriate disease-management measures."
    ],

    "Tomato___Leaf_Mold": [
        "Improve ventilation around tomato plants.",
        "Avoid unnecessary overhead watering.",
        "Remove severely affected leaves where practical.",
        "Monitor nearby plants regularly."
    ],

    "Tomato___healthy": [
        "The leaf appears healthy according to the AI prediction.",
        "Continue regular crop monitoring.",
        "Maintain proper irrigation and plant nutrition.",
        "Check new leaves regularly for symptoms."
    ]
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = next(iter(request.files.values()),None)

    if not file or file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    try:

        image = Image.open(file.stream).convert("RGB")
        image = image.resize((224, 224))

        # IMPORTANT:
        # The MobileNetV2 preprocessing layer is already inside
        # the converted model, so the image remains in 0-255 format.

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        interpreter.set_tensor(
            input_details["index"],
            image_array
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details["index"]
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[predicted_index] * 100
        )

        disease_code = class_names[predicted_index]

        crop = disease_code.split("_")[0]

        disease = (
            disease_code
            .replace("_", " ")
            .replace("_", " ")
        )

        advice = ADVICE.get(
            disease_code,
            []
        )

        return render_template(
            "index.html",
            result=True,
            crop=crop,
            disease=disease,
            confidence=confidence,
            advice=advice
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Prediction error: {e}"
        )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )