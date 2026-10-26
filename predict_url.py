
import json
import math
import numpy as np

from feature_extraction import extract_features

# Load the final Model A parameters
with open("phishguard_model_final.json", "r", encoding="utf-8") as file:
    model = json.load(file)

FEATURE_NAMES = model["feature_names"]


def predict_url(url):
    features = extract_features(url)

    # Keep the exact feature order used during training
    input_data = np.array(
        [features[name] for name in FEATURE_NAMES],
        dtype=float
    )

    # Apply the StandardScaler parameters
    mean = np.array(model["scaler_mean"])
    scale = np.array(model["scaler_scale"])
    coefficients = np.array(model["coefficients"][0])
    intercept = model["intercept"][0]

    scaled_data = (input_data - mean) / scale

    # Logistic Regression prediction
    score = np.dot(scaled_data, coefficients) + intercept
    probability = 1 / (1 + math.exp(-float(np.clip(score, -500, 500))))

    prediction = 1 if probability >= 0.5 else 0
    label = "Phishing" if prediction == 1 else "Legitimate"

    print("\nURL:", url)
    print("Prediction:", label)
    print(f"Phishing probability: {probability * 100:.2f}%")


if __name__ == "__main__":
    while True:
        url = input("\nEnter a URL (or type 'exit' to stop): ").strip()

        if url.lower() == "exit":
            break

        if not url:
            print("Please enter a URL.")
            continue

        try:
            predict_url(url)
        except Exception as error:
            print("Error:", error)
