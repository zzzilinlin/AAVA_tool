"""Model training nodes for AAVA - Dutch harmfulness classification."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix


DUTCH_STOPWORDS = [
    'de', 'het', 'een', 'en', 'van', 'in', 'is', 'op', 'te', 'dat', 'die', 'er',
    'voor', 'aan', 'met', 'als', 'zijn', 'ook', 'maar', 'om', 'niet', 'dan',
    'nog', 'wel', 'naar', 'kan', 'tot', 'bij', 'of', 'over', 'door', 'worden',
    'uit', 'al', 'zo', 'werd', 'heeft', 'haar', 'meer', 'zich', 'zou', 'tegen',
    'nu', 'wat', 'geen', 'dit', 'hun', 'was', 'hem', 'hebben', 'deze', 'zeer',
    'moet', 'worden', 'weer', 'ik', 'je', 'wij', 'zij', 'u', 'mij', 'jij', 'ons',
    'jullie', 'hen', 'zelf', 'daar', 'hier', 'waar', 'wanneer', 'hoe', 'waarom'
]


class TfidfModel:
    """TF-IDF + Logistic Regression model with optional categorical and numeric features."""
    
    def __init__(self, tfidf_vectorizer, label_encoder, cat_encoder, num_scaler, classifier, 
                 cat_columns, num_columns, text_column):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.label_encoder = label_encoder
        self.cat_encoder = cat_encoder
        self.num_scaler = num_scaler
        self.classifier = classifier
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.text_column = text_column
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = self._transform_features(data)
        predictions = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        X = self._transform_features(data)
        return self.classifier.predict_proba(X)
    
    def _transform_features(self, data: pd.DataFrame) -> csr_matrix:
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
        
        result = hstack(features_list)
        return result


class TfidfSoftLabelModel:
    """TF-IDF model trained on soft labels (probability distributions from pooled annotations)."""
    
    def __init__(self, tfidf_vectorizer, classes, classifier, text_column):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.classes = classes
        self.classifier = classifier
        self.text_column = text_column
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return the class with highest probability."""
        probs = self.predict_proba(data)
        predictions = np.argmax(probs, axis=1)
        return np.array([self.classes[i] for i in predictions])
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities for each class."""
        X = self.tfidf_vectorizer.transform(data[self.text_column].fillna(''))
        return self.classifier.predict_proba(X)
    
    def _transform_features(self, data: pd.DataFrame) -> csr_matrix:
        return self.tfidf_vectorizer.transform(data[self.text_column].fillna(''))


class SbertModel:
    """Sentence-BERT + Logistic Regression model with optional categorical and numeric features."""
    
    def __init__(self, sbert_model, label_encoder, cat_encoder, num_scaler, classifier,
                 cat_columns, num_columns, text_column, batch_size=64):
        self.sbert_model = sbert_model
        self.label_encoder = label_encoder
        self.cat_encoder = cat_encoder
        self.num_scaler = num_scaler
        self.classifier = classifier
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.text_column = text_column
        self.batch_size = batch_size
        self._device = self._detect_device()
    
    def _detect_device(self) -> str:
        """Detect best available device (CUDA GPU or CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = self._transform_features(data)
        predictions = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        X = self._transform_features(data)
        return self.classifier.predict_proba(X)
    
    def _transform_features(self, data: pd.DataFrame) -> np.ndarray:
        texts = data[self.text_column].fillna('').tolist()
        text_embeddings = self.sbert_model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            device=self._device,
        )
        
        features_list = [text_embeddings]
        
        if self.cat_columns and self.cat_encoder is not None:
            cat_data = data[self.cat_columns].fillna('unknown')
            cat_features = self.cat_encoder.transform(cat_data)
            if hasattr(cat_features, 'toarray'):
                cat_features = cat_features.toarray()
            features_list.append(cat_features)
        
        if self.num_columns and self.num_scaler is not None:
            num_data = data[self.num_columns].fillna(0).values
            num_features = self.num_scaler.transform(num_data)
            features_list.append(num_features)
        
        return np.hstack(features_list)


def train_tfidf_model(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    logreg_params: Dict[str, Any],
    text_column: str = "sentence",
    label_column: str = "harmfulness_level",
    cat_columns: Optional[List[str]] = None,
    num_columns: Optional[List[str]] = None,
    model_type: str = "general",
) -> Dict[str, Any]:
    """
    Train TF-IDF + Logistic Regression model.
    
    Args:
        train_data: Training DataFrame
        tfidf_params: TF-IDF vectorizer parameters
        logreg_params: Logistic Regression parameters
        text_column: Name of text column
        label_column: Name of label column
        cat_columns: List of categorical column names (only used for feature-aware model)
        num_columns: List of numeric column names (only used for feature-aware model)
        model_type: "general" (text only) or "feature_aware" (text + demographics)
    
    Returns:
        Dictionary with trained model and metadata
    """
    if model_type == "general":
        cat_columns = []
        num_columns = []
    else:
        cat_columns = cat_columns or []
        num_columns = num_columns or []
    
    ngram_range = tfidf_params.get('ngram_range', (1, 2))
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)
    
    tfidf = TfidfVectorizer(
        max_features=tfidf_params.get('max_features', 5000),
        ngram_range=ngram_range,
        min_df=tfidf_params.get('min_df', 2),
        stop_words=DUTCH_STOPWORDS,
        lowercase=True,
        strip_accents='unicode',
    )
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_data[label_column].values)
    
    text_features = tfidf.fit_transform(train_data[text_column].fillna(''))
    
    cat_encoder = None
    cat_features = None
    if cat_columns:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        cat_data = train_data[cat_columns].fillna('unknown')
        cat_features = cat_encoder.fit_transform(cat_data)
    
    num_scaler = None
    num_features = None
    if num_columns:
        num_scaler = StandardScaler()
        num_data = train_data[num_columns].fillna(0).values
        num_features = csr_matrix(num_scaler.fit_transform(num_data))
    
    features_list = [text_features]
    if cat_features is not None:
        features_list.append(cat_features)
    if num_features is not None:
        features_list.append(num_features)
    
    X = hstack(features_list)
    
    classifier = LogisticRegression(**logreg_params)
    classifier.fit(X, y)
    
    model = TfidfModel(
        tfidf_vectorizer=tfidf,
        label_encoder=label_encoder,
        cat_encoder=cat_encoder,
        num_scaler=num_scaler,
        classifier=classifier,
        cat_columns=cat_columns,
        num_columns=num_columns,
        text_column=text_column,
    )
    
    n_text_features = text_features.shape[1]
    n_cat_features = cat_features.shape[1] if cat_features is not None else 0
    n_num_features = len(num_columns)
    
    return {
        "model": model,
        "model_name": f"tfidf_{model_type}",
        "model_type": model_type,
        "tfidf_params": tfidf_params,
        "logreg_params": logreg_params,
        "n_samples_trained": len(train_data),
        "n_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "n_text_features": n_text_features,
        "n_cat_features": n_cat_features,
        "n_num_features": n_num_features,
        "total_features": n_text_features + n_cat_features + n_num_features,
    }


def train_tfidf_soft_label_model(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    logreg_params: Dict[str, Any],
    text_column: str = "sentence",
    label_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train TF-IDF model using soft labels (pooled annotation distributions).
    
    This model is trained on probability distributions rather than hard labels,
    using sample weighting to approximate cross-entropy loss on soft targets.
    Each sample contributes to training proportionally to the soft label distribution.
    
    Args:
        train_data: Training DataFrame with soft_* columns for each class
        tfidf_params: TF-IDF vectorizer parameters
        logreg_params: Logistic Regression parameters
        text_column: Name of text column
        label_classes: List of label class names (e.g., ['high', 'low', 'medium'])
    
    Returns:
        Dictionary with trained model and metadata
    """
    if label_classes is None:
        soft_cols = [c for c in train_data.columns if c.startswith('soft_')]
        label_classes = sorted([c.replace('soft_', '') for c in soft_cols])
    
    soft_columns = [f'soft_{cls}' for cls in label_classes]
    
    for col in soft_columns:
        if col not in train_data.columns:
            raise ValueError(f"Missing soft label column: {col}. Run pool_annotations_to_soft_labels first.")
    
    print(f"Training TF-IDF Soft Label model on {len(train_data)} samples")
    print(f"Label classes: {label_classes}")
    
    ngram_range = tfidf_params.get('ngram_range', (1, 2))
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)
    
    tfidf = TfidfVectorizer(
        max_features=tfidf_params.get('max_features', 5000),
        ngram_range=ngram_range,
        min_df=tfidf_params.get('min_df', 2),
        stop_words=DUTCH_STOPWORDS,
        lowercase=True,
        strip_accents='unicode',
    )
    
    X_orig = tfidf.fit_transform(train_data[text_column].fillna(''))
    y_soft = train_data[soft_columns].values
    
    texts_expanded = []
    labels_expanded = []
    weights_expanded = []
    
    for idx in range(len(train_data)):
        for class_idx, cls in enumerate(label_classes):
            prob = y_soft[idx, class_idx]
            if prob > 0:
                texts_expanded.append(idx)
                labels_expanded.append(class_idx)
                weights_expanded.append(prob)
    
    from scipy.sparse import vstack as sparse_vstack
    X_expanded = sparse_vstack([X_orig[i] for i in texts_expanded])
    y_expanded = np.array(labels_expanded)
    sample_weights = np.array(weights_expanded)
    
    print(f"Expanded training set: {len(y_expanded)} weighted samples from {len(train_data)} originals")
    
    logreg_params_copy = logreg_params.copy()
    logreg_params_copy.pop('class_weight', None)
    
    classifier = LogisticRegression(**logreg_params_copy)
    classifier.fit(X_expanded, y_expanded, sample_weight=sample_weights)
    
    model = TfidfSoftLabelModel(
        tfidf_vectorizer=tfidf,
        classes=label_classes,
        classifier=classifier,
        text_column=text_column,
    )
    
    n_text_features = X_orig.shape[1]
    
    return {
        "model": model,
        "model_name": "tfidf_general_soft",
        "model_type": "general_soft",
        "tfidf_params": tfidf_params,
        "logreg_params": logreg_params,
        "n_samples_trained": len(train_data),
        "n_expanded_samples": len(y_expanded),
        "n_classes": len(label_classes),
        "classes": label_classes,
        "n_text_features": n_text_features,
        "n_cat_features": 0,
        "n_num_features": 0,
        "total_features": n_text_features,
        "uses_soft_labels": True,
    }


def train_sbert_model(
    train_data: pd.DataFrame,
    sbert_params: Dict[str, Any],
    logreg_params: Dict[str, Any],
    text_column: str = "sentence",
    label_column: str = "harmfulness_level",
    cat_columns: Optional[List[str]] = None,
    num_columns: Optional[List[str]] = None,
    model_type: str = "general",
) -> Dict[str, Any]:
    """
    Train Sentence-BERT + Logistic Regression model.
    
    Args:
        train_data: Training DataFrame
        sbert_params: Sentence-BERT parameters
        logreg_params: Logistic Regression parameters
        text_column: Name of text column
        label_column: Name of label column
        cat_columns: List of categorical column names (only used for feature-aware model)
        num_columns: List of numeric column names (only used for feature-aware model)
        model_type: "general" (text only) or "feature_aware" (text + demographics)
    
    Returns:
        Dictionary with trained model and metadata
    """
    from sentence_transformers import SentenceTransformer
    
    if model_type == "general":
        cat_columns = []
        num_columns = []
    else:
        cat_columns = cat_columns or []
        num_columns = num_columns or []
    
    model_name = sbert_params.get('model_name', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    batch_size = sbert_params.get('batch_size', 64)
    
    print(f"Loading SBERT model: {model_name}")
    sbert_model = SentenceTransformer(model_name)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_data[label_column].values)
    
    print(f"Encoding {len(train_data)} sentences with SBERT...")
    texts = train_data[text_column].fillna('').tolist()
    text_embeddings = sbert_model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    cat_encoder = None
    cat_features = None
    if cat_columns:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_data = train_data[cat_columns].fillna('unknown')
        cat_features = cat_encoder.fit_transform(cat_data)
    
    num_scaler = None
    num_features = None
    if num_columns:
        num_scaler = StandardScaler()
        num_data = train_data[num_columns].fillna(0).values
        num_features = num_scaler.fit_transform(num_data)
    
    features_list = [text_embeddings]
    if cat_features is not None:
        features_list.append(cat_features)
    if num_features is not None:
        features_list.append(num_features)
    
    X = np.hstack(features_list)
    
    print(f"Training Logistic Regression on {X.shape[1]} features...")
    classifier = LogisticRegression(**logreg_params)
    classifier.fit(X, y)
    
    model = SbertModel(
        sbert_model=sbert_model,
        label_encoder=label_encoder,
        cat_encoder=cat_encoder,
        num_scaler=num_scaler,
        classifier=classifier,
        cat_columns=cat_columns,
        num_columns=num_columns,
        text_column=text_column,
        batch_size=batch_size,
    )
    
    n_text_features = text_embeddings.shape[1]
    n_cat_features = cat_features.shape[1] if cat_features is not None else 0
    n_num_features = len(num_columns)
    
    return {
        "model": model,
        "model_name": f"sbert_{model_type}",
        "model_type": model_type,
        "sbert_params": sbert_params,
        "logreg_params": logreg_params,
        "n_samples_trained": len(train_data),
        "n_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "n_text_features": n_text_features,
        "n_cat_features": n_cat_features,
        "n_num_features": n_num_features,
        "total_features": n_text_features + n_cat_features + n_num_features,
        "embedding_dim": n_text_features,
    }


def train_all_models(
    train_data: pd.DataFrame,
    tfidf_params: Dict[str, Any],
    logreg_params: Dict[str, Any],
    sbert_params: Dict[str, Any],
    text_column: str = "sentence",
    label_column: str = "harmfulness_level",
    cat_columns: Optional[List[str]] = None,
    num_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train all classifier models for comparison.
    
    Trains 4 models:
    1. TF-IDF General (text only)
    2. TF-IDF Feature-Aware (text + demographics)
    3. SBERT General (text only)
    4. SBERT Feature-Aware (text + demographics)
    
    Args:
        train_data: Training DataFrame
        tfidf_params: TF-IDF vectorizer parameters
        logreg_params: Logistic Regression parameters
        sbert_params: Sentence-BERT parameters
        text_column: Name of text column
        label_column: Name of label column
        cat_columns: List of categorical column names
        num_columns: List of numeric column names
    
    Returns:
        Dictionary containing all trained models
    """
    models = {}
    
    if cat_columns is None:
        cat_columns = ['gender', 'education_level', 'ethnicity', 'nationality']
    if num_columns is None:
        num_columns = ['age']
    
    print("=" * 60)
    print("Training TF-IDF General Model (text only)...")
    print("=" * 60)
    models["tfidf_general"] = train_tfidf_model(
        train_data, tfidf_params, logreg_params, 
        text_column, label_column, cat_columns, num_columns,
        model_type="general"
    )
    print(f"TF-IDF General: {models['tfidf_general']['total_features']} text features")
    
    print()
    print("=" * 60)
    print("Training TF-IDF Feature-Aware Model (text + demographics)...")
    print("=" * 60)
    models["tfidf_feature_aware"] = train_tfidf_model(
        train_data, tfidf_params, logreg_params,
        text_column, label_column, cat_columns, num_columns,
        model_type="feature_aware"
    )
    print(f"TF-IDF Feature-Aware: {models['tfidf_feature_aware']['total_features']} total features")
    print(f"  - Text: {models['tfidf_feature_aware']['n_text_features']}, Cat: {models['tfidf_feature_aware']['n_cat_features']}, Num: {models['tfidf_feature_aware']['n_num_features']}")
    
    print()
    print("=" * 60)
    print("Training SBERT General Model (text only)...")
    print("=" * 60)
    try:
        models["sbert_general"] = train_sbert_model(
            train_data, sbert_params, logreg_params,
            text_column, label_column, cat_columns, num_columns,
            model_type="general"
        )
        print(f"SBERT General: {models['sbert_general']['total_features']} features (embedding dim: {models['sbert_general']['embedding_dim']})")
    except Exception as e:
        print(f"Warning: SBERT General training failed: {e}")
    
    print()
    print("=" * 60)
    print("Training SBERT Feature-Aware Model (text + demographics)...")
    print("=" * 60)
    try:
        models["sbert_feature_aware"] = train_sbert_model(
            train_data, sbert_params, logreg_params,
            text_column, label_column, cat_columns, num_columns,
            model_type="feature_aware"
        )
        print(f"SBERT Feature-Aware: {models['sbert_feature_aware']['total_features']} total features")
        print(f"  - Text: {models['sbert_feature_aware']['n_text_features']}, Cat: {models['sbert_feature_aware']['n_cat_features']}, Num: {models['sbert_feature_aware']['n_num_features']}")
    except Exception as e:
        print(f"Warning: SBERT Feature-Aware training failed: {e}")
    
    print()
    print(f"Successfully trained {len(models)} models.")
    print(f"Classes: {models['tfidf_general']['classes']}")
    return models
