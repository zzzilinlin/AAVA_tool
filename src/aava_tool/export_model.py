"""Export trained models for frontend integration."""
import pickle
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def export_model(
    model_path: str = "data/06_models/best_model.pkl",
    export_dir: str = "exported_model",
    include_all_models: bool = False,
) -> Dict[str, Any]:
    """
    Export trained model(s) for use in external applications.
    
    Args:
        model_path: Path to the best model pickle file
        export_dir: Directory to export the model to
        include_all_models: If True, also export all trained models
    
    Returns:
        Dictionary with export information
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "rb") as f:
        best_model_data = pickle.load(f)
    
    model = best_model_data["model"]
    model_name = best_model_data.get("model_name", "unknown")
    model_type = best_model_data.get("model_type", "unknown")
    
    model_export_path = export_path / "model.pkl"
    with open(model_export_path, "wb") as f:
        pickle.dump(best_model_data, f)
    
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "classes": best_model_data.get("classes", []),
        "n_classes": best_model_data.get("n_classes", 0),
        "n_text_features": best_model_data.get("n_text_features", 0),
        "n_cat_features": best_model_data.get("n_cat_features", 0),
        "n_num_features": best_model_data.get("n_num_features", 0),
        "total_features": best_model_data.get("total_features", 0),
        "text_column": model.text_column,
        "cat_columns": model.cat_columns if hasattr(model, 'cat_columns') else [],
        "num_columns": model.num_columns if hasattr(model, 'num_columns') else [],
        "accuracy": best_model_data.get("accuracy", None),
        "f1_weighted": best_model_data.get("f1_weighted", None),
    }
    
    metadata_path = export_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    predictor_code = '''"""Standalone predictor for AAVA harmfulness classification model."""
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


class HarmfulnessPredictor:
    """
    Standalone predictor for Dutch text harmfulness classification.
    
    Usage:
        predictor = HarmfulnessPredictor("path/to/exported_model")
        
        # Simple prediction (text only)
        result = predictor.predict("Dit is een zin om te classificeren.")
        
        # Batch prediction
        results = predictor.predict_batch(["Zin 1", "Zin 2", "Zin 3"])
        
        # Feature-aware prediction (if model supports it)
        result = predictor.predict(
            "Dit is een zin.",
            demographics={"gender": "male", "age": 30, "education_level": "HBO"}
        )
    """
    
    def __init__(self, model_dir: str):
        """
        Load the exported model.
        
        Args:
            model_dir: Path to the exported model directory
        """
        model_path = Path(model_dir)
        
        with open(model_path / "model.pkl", "rb") as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data["model"]
        self.model_name = self.model_data.get("model_name", "unknown")
        self.model_type = self.model_data.get("model_type", "unknown")
        self.classes = self.model_data.get("classes", [])
        
        self.text_column = self.model.text_column
        self.cat_columns = getattr(self.model, 'cat_columns', []) or []
        self.num_columns = getattr(self.model, 'num_columns', []) or []
        
        self.is_feature_aware = len(self.cat_columns) > 0 or len(self.num_columns) > 0
    
    def predict(
        self, 
        text: str, 
        demographics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict harmfulness level for a single text.
        
        Args:
            text: The Dutch text to classify
            demographics: Optional dict with demographic features
                         (gender, age, education_level, ethnicity, nationality)
        
        Returns:
            Dictionary with prediction results
        """
        data = {self.text_column: [text]}
        
        if demographics and self.is_feature_aware:
            for col in self.cat_columns:
                data[col] = [demographics.get(col, "unknown")]
            for col in self.num_columns:
                data[col] = [demographics.get(col, 0)]
        elif self.is_feature_aware:
            for col in self.cat_columns:
                data[col] = ["unknown"]
            for col in self.num_columns:
                data[col] = [0]
        
        df = pd.DataFrame(data)
        
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes, probabilities)}
        
        return {
            "text": text,
            "prediction": prediction,
            "confidence": float(max(probabilities)),
            "probabilities": prob_dict,
            "model_type": self.model_type,
        }
    
    def predict_batch(
        self, 
        texts: List[str],
        demographics_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict harmfulness levels for multiple texts.
        
        Args:
            texts: List of Dutch texts to classify
            demographics_list: Optional list of demographic dicts (one per text)
        
        Returns:
            List of prediction dictionaries
        """
        data = {self.text_column: texts}
        
        if demographics_list and self.is_feature_aware:
            for col in self.cat_columns:
                data[col] = [d.get(col, "unknown") for d in demographics_list]
            for col in self.num_columns:
                data[col] = [d.get(col, 0) for d in demographics_list]
        elif self.is_feature_aware:
            for col in self.cat_columns:
                data[col] = ["unknown"] * len(texts)
            for col in self.num_columns:
                data[col] = [0] * len(texts)
        
        df = pd.DataFrame(data)
        
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)
        
        results = []
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
            results.append({
                "text": text,
                "prediction": pred,
                "confidence": float(max(probs)),
                "probabilities": prob_dict,
                "model_type": self.model_type,
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "is_feature_aware": self.is_feature_aware,
            "classes": self.classes,
            "text_column": self.text_column,
            "cat_columns": self.cat_columns,
            "num_columns": self.num_columns,
            "accuracy": self.model_data.get("accuracy"),
            "f1_weighted": self.model_data.get("f1_weighted"),
        }


if __name__ == "__main__":
    import sys
    
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    predictor = HarmfulnessPredictor(model_dir)
    
    print("Model Info:")
    print(predictor.get_model_info())
    print()
    
    test_text = "Dit is een test zin voor classificatie."
    result = predictor.predict(test_text)
    print(f"Test prediction for: '{test_text}'")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: {result['probabilities']}")
'''
    
    predictor_path = export_path / "predictor.py"
    with open(predictor_path, "w") as f:
        f.write(predictor_code)
    
    readme_content = f"""# AAVA Harmfulness Classification Model

## Model Information
- **Model Name**: {model_name}
- **Model Type**: {model_type}
- **Classes**: {', '.join(metadata['classes'])}

## Files
- `model.pkl` - The trained model (pickle format)
- `metadata.json` - Model metadata and configuration
- `predictor.py` - Standalone prediction script

## Requirements
```
pandas
scikit-learn
{"sentence-transformers" if "sbert" in model_name else ""}
```

## Usage

### Python
```python
from predictor import HarmfulnessPredictor

# Load the model
predictor = HarmfulnessPredictor("path/to/exported_model")

# Simple prediction
result = predictor.predict("Dit is een Nederlandse zin.")
print(result['prediction'])  # 'low', 'medium', or 'high'
print(result['confidence'])  # e.g., 0.85

# Batch prediction
results = predictor.predict_batch([
    "Eerste zin",
    "Tweede zin",
    "Derde zin"
])
```

### With Demographics (Feature-Aware Models)
```python
result = predictor.predict(
    "Dit is een zin.",
    demographics={{
        "gender": "male",
        "age": 35,
        "education_level": "HBO",
        "ethnicity": "Dutch",
        "nationality": "Netherlands"
    }}
)
```

## Integration Notes
- The model expects Dutch text in the `sentence` field
- For feature-aware models, demographic features improve predictions
- If demographics are not provided, defaults are used
"""
    
    readme_path = export_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    if include_all_models:
        all_models_path = Path("data/06_models/trained_models.pkl")
        if all_models_path.exists():
            shutil.copy(all_models_path, export_path / "all_models.pkl")
    
    print(f"\nModel exported successfully to: {export_path.absolute()}")
    print(f"\nExported files:")
    for file in export_path.iterdir():
        print(f"  - {file.name}")
    
    return {
        "export_dir": str(export_path.absolute()),
        "model_name": model_name,
        "model_type": model_type,
        "files": [f.name for f in export_path.iterdir()],
    }


if __name__ == "__main__":
    export_model()
