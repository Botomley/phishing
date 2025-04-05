from flask import Flask, request, jsonify
import joblib
import pandas as pd
import validators
import os

app = Flask(__name__)

# Load your ML pipeline
pipeline_path = os.path.abspath("C:/Users/Admin/phishing/phishing-detector/pipeline.pkl")

try:
    pipeline = joblib.load(pipeline_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading pipeline: {str(e)}")
    pipeline = None

def extract_features(url):
    # Replace with your actual feature extraction logic
    return {
        "URLLength": len(url),
        "DomainLength": len(url.split("//")[-1].split("/")[0]),
        "TLD": url.split(".")[-1]
    }

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json.get("url", "")
    if not validators.url(url):
        return jsonify({"error": "Invalid URL"}), 400
        
    features = pd.DataFrame([extract_features(url)])
    if pipeline:
        prediction = pipeline.predict(features)[0]
        return jsonify({
            "prediction": "phishing" if prediction == 1 else "legitimate",
            "confidence": float(pipeline.predict_proba(features)[0][1])
        })
    else:
        return jsonify({"error": "Model not loaded"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)