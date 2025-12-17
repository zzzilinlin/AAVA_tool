"""Flask API for AAVA Harmfulness Classification models."""
import os
import pickle
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np


app = Flask(__name__)
CORS(app)

MODELS_DIR = Path(__file__).parent.parent / "data" / "06_models"

models = {}


class TfidfSoftLabelModel:
    """TF-IDF model trained on soft labels (required for unpickling)."""
    
    def __init__(self, tfidf_vectorizer, classes, classifier, text_column):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.classes = classes
        self.classifier = classifier
        self.text_column = text_column
    
    def predict(self, data):
        probs = self.predict_proba(data)
        predictions = np.argmax(probs, axis=1)
        return np.array([self.classes[i] for i in predictions])
    
    def predict_proba(self, data):
        X = self.tfidf_vectorizer.transform(data[self.text_column].fillna(''))
        return self.classifier.predict_proba(X)


class TfidfModel:
    """TF-IDF + Logistic Regression model (required for unpickling)."""
    
    def __init__(self, tfidf_vectorizer, label_encoder, cat_encoder, num_scaler, 
                 classifier, cat_columns, num_columns, text_column):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.label_encoder = label_encoder
        self.cat_encoder = cat_encoder
        self.num_scaler = num_scaler
        self.classifier = classifier
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.text_column = text_column
    
    def predict(self, data):
        X = self._transform_features(data)
        predictions = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, data):
        X = self._transform_features(data)
        return self.classifier.predict_proba(X)
    
    def _transform_features(self, data):
        from scipy.sparse import hstack, csr_matrix
        text_features = self.tfidf_vectorizer.transform(data[self.text_column].fillna(''))
        
        features_list = [text_features]
        
        if self.cat_columns and self.cat_encoder is not None:
            cat_data = data[self.cat_columns].fillna('unknown')
            cat_features = self.cat_encoder.transform(cat_data)
            features_list.append(cat_features)
        
        if self.num_columns and self.num_scaler is not None:
            num_data = data[self.num_columns].fillna(0).values
            num_features = csr_matrix(self.num_scaler.transform(num_data))
            features_list.append(num_features)
        
        return hstack(features_list)


def load_models():
    """Load all available models on startup."""
    global models
    
    model_dirs = {
        "general_soft": MODELS_DIR / "tfidf_general_soft",
        "general": MODELS_DIR / "tfidf_general",
        "feature_aware": MODELS_DIR / "tfidf_feature_aware",
    }
    
    for name, path in model_dirs.items():
        model_file = path / "model.pkl"
        if model_file.exists():
            try:
                with open(model_file, "rb") as f:
                    model_data = pickle.load(f)
                models[name] = {
                    "model": model_data["model"],
                    "metadata": {
                        "model_name": model_data.get("model_name", name),
                        "model_type": model_data.get("model_type", name),
                        "classes": model_data.get("classes", []),
                        "n_features": model_data.get("total_features", 0),
                        "uses_soft_labels": model_data.get("uses_soft_labels", False),
                    }
                }
                print(f"Loaded model: {name}")
            except Exception as e:
                print(f"Failed to load model {name}: {e}")
    
    print(f"Loaded {len(models)} models: {list(models.keys())}")


@app.route("/")
def index():
    """API information endpoint."""
    return jsonify({
        "name": "AAVA Harmfulness Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/models": "GET - List available models",
            "/predict/general": "POST - Predict using general (text-only) model",
            "/predict/feature-aware": "POST - Predict using feature-aware model",
            "/predict/batch": "POST - Batch prediction",
        },
        "models_loaded": list(models.keys()),
    })


@app.route("/models", methods=["GET"])
def list_models():
    """List available models and their metadata."""
    return jsonify({
        "models": {
            name: data["metadata"] for name, data in models.items()
        }
    })


@app.route("/predict/general", methods=["POST"])
def predict_general():
    """
    Predict harmfulness using the general (text-only) model.
    
    Request body:
    {
        "text": "Dutch text to classify"
    }
    OR
    {
        "texts": ["text1", "text2", ...]
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    if "text" in data:
        texts = [data["text"]]
    elif "texts" in data:
        texts = data["texts"]
    else:
        return jsonify({"error": "Provide 'text' or 'texts' in request body"}), 400
    
    return _do_general_prediction(texts)


@app.route("/predict/feature-aware", methods=["POST"])
def predict_feature_aware():
    """
    Predict harmfulness using the feature-aware model (text + demographics).
    
    Request body:
    {
        "text": "Dutch text to classify",
        "gender": "male|female|other",
        "age": 25,
        "education_level": "high|medium|low",
        "ethnicity": "...",
        "nationality": "..."
    }
    OR for batch:
    {
        "items": [
            {"text": "...", "gender": "...", ...},
            ...
        ]
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    if "items" in data:
        items = data["items"]
    elif "text" in data:
        items = [data]
    else:
        return jsonify({"error": "Provide 'text' with demographics or 'items' array"}), 400
    
    return _do_feature_aware_prediction(items)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint.
    
    Request body:
    {
        "model": "general" | "feature_aware",
        "texts": ["text1", "text2", ...],
        "demographics": [...] (optional, for feature_aware)
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    model_type = data.get("model", "general")
    texts = data.get("texts", [])
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    
    if model_type == "feature_aware":
        demographics = data.get("demographics", [])
        items = []
        for i, text in enumerate(texts):
            item = {"text": text}
            if i < len(demographics):
                item.update(demographics[i])
            items.append(item)
        return _do_feature_aware_prediction(items)
    else:
        return _do_general_prediction(texts)


def _do_general_prediction(texts):
    """Internal helper for general model prediction."""
    model_key = "general_soft" if "general_soft" in models else "general"
    
    if model_key not in models:
        return jsonify({"error": "General model not loaded"}), 500
    
    model = models[model_key]["model"]
    metadata = models[model_key]["metadata"]
    
    df = pd.DataFrame({"sentence": texts})
    
    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        classes = metadata["classes"]
        
        results = []
        for text, pred, probs in zip(texts, predictions, probabilities):
            prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
            results.append({
                "text": text,
                "prediction": str(pred),
                "confidence": float(max(probs)),
                "probabilities": prob_dict,
            })
        
        if len(results) == 1:
            return jsonify({"result": results[0], "model": model_key})
        
        return jsonify({"results": results, "model": model_key})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _do_feature_aware_prediction(items):
    """Internal helper for feature-aware model prediction."""
    if "feature_aware" not in models:
        return jsonify({"error": "Feature-aware model not loaded"}), 500
    
    df = pd.DataFrame(items)
    df = df.rename(columns={"text": "sentence"})
    
    required_cols = ["sentence", "gender", "age", "education_level", "ethnicity", "nationality"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "unknown" if col != "age" else 30
    
    model = models["feature_aware"]["model"]
    metadata = models["feature_aware"]["metadata"]
    
    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        classes = metadata["classes"]
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
            results.append({
                "text": items[i].get("text", ""),
                "prediction": str(pred),
                "confidence": float(max(probs)),
                "probabilities": prob_dict,
                "demographics_used": {
                    "gender": str(df.iloc[i].get("gender", "unknown")),
                    "age": int(df.iloc[i].get("age", 30)),
                    "education_level": str(df.iloc[i].get("education_level", "unknown")),
                    "ethnicity": str(df.iloc[i].get("ethnicity", "unknown")),
                    "nationality": str(df.iloc[i].get("nationality", "unknown")),
                }
            })
        
        if len(results) == 1:
            return jsonify({"result": results[0], "model": "feature_aware"})
        
        return jsonify({"results": results, "model": "feature_aware"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models),
    })


load_models()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
