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


def train_all_models(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    logistic_params: Dict[str, Any],
    random_forest_params: Dict[str, Any],
    gradient_boosting_params: Dict[str, Any],
    svm_params: Dict[str, Any],
    naive_bayes_params: Dict[str, Any],
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
    
    print(f"Successfully trained {len(models)} models.")
    return models
