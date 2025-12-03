"""Model training nodes for AAVA."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline as SklearnPipeline


LABEL_NAMES = {
    0: "Not Harmful",
    1: "Slightly Harmful",
    2: "Harmful",
    3: "Very Harmful",
}

LABEL_NAMES_LIST = ["Not Harmful", "Slightly Harmful", "Harmful", "Very Harmful"]


class SetFitWrapper:
    """Wrapper to make SetFit model compatible with sklearn-style predict."""
    def __init__(self, setfit_model, label_map):
        self.setfit_model = setfit_model
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        predictions = self.setfit_model.predict(list(texts))
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        numeric_predictions = []
        for pred in predictions:
            if isinstance(pred, str):
                numeric_predictions.append(self.reverse_label_map.get(pred, 0))
            else:
                numeric_predictions.append(pred)
        return np.array(numeric_predictions)
    
    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.setfit_model.predict_proba(list(texts))


def create_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with specified parameters."""
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=None,
        lowercase=True,
        strip_accents="unicode",
    )


def train_logistic_regression(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    model_params: Dict[str, Any],
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Train a Logistic Regression classifier.
    
    Args:
        train_data: Training DataFrame.
        tfidf_params: Parameters for TF-IDF vectorizer.
        model_params: Parameters for the classifier.
        text_column: Name of text column.
        label_column: Name of label column.
    
    Returns:
        Dictionary containing the trained model and metadata.
    """
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    classifier = LogisticRegression(**model_params)
    
    pipeline = SklearnPipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    
    X = train_data[text_column].values
    y = train_data[label_column].values
    
    pipeline.fit(X, y)
    
    return {
        "model": pipeline,
        "model_name": "logistic_regression",
        "tfidf_params": tfidf_params,
        "model_params": model_params,
        "n_samples_trained": len(train_data),
    }


def train_random_forest(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    model_params: Dict[str, Any],
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """Train a Random Forest classifier."""
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    classifier = RandomForestClassifier(**model_params)
    
    pipeline = SklearnPipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    
    X = train_data[text_column].values
    y = train_data[label_column].values
    
    pipeline.fit(X, y)
    
    return {
        "model": pipeline,
        "model_name": "random_forest",
        "tfidf_params": tfidf_params,
        "model_params": model_params,
        "n_samples_trained": len(train_data),
    }


def train_gradient_boosting(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    model_params: Dict[str, Any],
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """Train a Gradient Boosting classifier."""
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    classifier = GradientBoostingClassifier(**model_params)
    
    pipeline = SklearnPipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    
    X = train_data[text_column].values
    y = train_data[label_column].values
    
    pipeline.fit(X, y)
    
    return {
        "model": pipeline,
        "model_name": "gradient_boosting",
        "tfidf_params": tfidf_params,
        "model_params": model_params,
        "n_samples_trained": len(train_data),
    }


def train_svm(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    model_params: Dict[str, Any],
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """Train an SVM classifier."""
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    classifier = SVC(**model_params)
    
    pipeline = SklearnPipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    
    X = train_data[text_column].values
    y = train_data[label_column].values
    
    pipeline.fit(X, y)
    
    return {
        "model": pipeline,
        "model_name": "svm",
        "tfidf_params": tfidf_params,
        "model_params": model_params,
        "n_samples_trained": len(train_data),
    }


def train_naive_bayes(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    model_params: Dict[str, Any],
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """Train a Naive Bayes classifier."""
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    classifier = MultinomialNB(**model_params)
    
    pipeline = SklearnPipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    
    X = train_data[text_column].values
    y = train_data[label_column].values
    
    pipeline.fit(X, y)
    
    return {
        "model": pipeline,
        "model_name": "naive_bayes",
        "tfidf_params": tfidf_params,
        "model_params": model_params,
        "n_samples_trained": len(train_data),
    }


def train_setfit(
    train_data: pd.DataFrame,
    model_params: Dict[str, Any],
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Train a SetFit classifier using sentence transformers.
    
    SetFit is a few-shot learning framework that:
    - Uses sentence embeddings for efficient text classification
    - Works well with small datasets (8-16 samples per class)
    - Doesn't require prompts like GPT-3/T0
    
    Args:
        train_data: Training DataFrame.
        model_params: Parameters for SetFit model and training.
        text_column: Name of text column.
        label_column: Name of label column.
    
    Returns:
        Dictionary containing the trained model and metadata.
    """
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    
    model_name = model_params.get("model_name", "sentence-transformers/paraphrase-mpnet-base-v2")
    batch_size = model_params.get("batch_size", 16)
    num_epochs = model_params.get("num_epochs", 1)
    num_iterations = model_params.get("num_iterations", 20)
    
    X = train_data[text_column].tolist()
    y = train_data[label_column].tolist()
    
    train_dataset = Dataset.from_dict({
        "text": X,
        "label": y
    })
    
    unique_labels = sorted(train_data[label_column].unique())
    labels = [LABEL_NAMES.get(lbl, f"Label_{lbl}") for lbl in unique_labels]
    
    model = SetFitModel.from_pretrained(
        model_name,
        labels=labels,
    )
    
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        column_mapping={"text": "text", "label": "label"},
    )
    
    trainer.train()
    
    label_map = {i: LABEL_NAMES.get(i, f"Label_{i}") for i in unique_labels}
    wrapped_model = SetFitWrapper(model, label_map)
    
    return {
        "model": wrapped_model,
        "setfit_model": model,
        "model_name": "setfit",
        "model_params": model_params,
        "n_samples_trained": len(train_data),
        "base_model": model_name,
    }


def train_all_models(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    logistic_params: Dict[str, Any],
    random_forest_params: Dict[str, Any],
    gradient_boosting_params: Dict[str, Any],
    svm_params: Dict[str, Any],
    naive_bayes_params: Dict[str, Any],
    setfit_params: Dict[str, Any],
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Train all classifier models for comparison.
    
    Returns:
        Dictionary containing all trained models.
    """
    models = {}
    
    print("Training Logistic Regression...")
    models["logistic_regression"] = train_logistic_regression(
        train_data, tfidf_params, logistic_params, text_column, label_column
    )
    
    print("Training Random Forest...")
    models["random_forest"] = train_random_forest(
        train_data, tfidf_params, random_forest_params, text_column, label_column
    )
    
    print("Training Gradient Boosting...")
    models["gradient_boosting"] = train_gradient_boosting(
        train_data, tfidf_params, gradient_boosting_params, text_column, label_column
    )
    
    print("Training SVM...")
    models["svm"] = train_svm(
        train_data, tfidf_params, svm_params, text_column, label_column
    )
    
    print("Training Naive Bayes...")
    models["naive_bayes"] = train_naive_bayes(
        train_data, tfidf_params, naive_bayes_params, text_column, label_column
    )
    
    print("Training SetFit (few-shot learning)...")
    try:
        models["setfit"] = train_setfit(
            train_data, setfit_params, text_column, label_column
        )
    except Exception as e:
        print(f"Warning: SetFit training failed: {e}")
        print("Continuing with other models...")
    
    print(f"Successfully trained {len(models)} models.")
    return models
