"""Data processing nodes for AAVA Harmfulness Classification."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"['']", "'", text)
    
    return text


def preprocess_data(
    dummy_data: pd.DataFrame,
    sample_size: int = None,
) -> pd.DataFrame:
    """
    Preprocess the harmfulness data for model training.
    
    Args:
        dummy_data: DataFrame with columns: sentence, article, harmfulness_level, 
                   gender, age, education_level, ethnicity, nationality
        sample_size: Optional number of rows to sample for testing (None = use all data).
    
    Returns:
        Preprocessed DataFrame.
    """
    df = dummy_data.copy()
    
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    if 'sentence' in df.columns:
        df['sentence'] = df['sentence'].apply(clean_text)
    
    if 'article' in df.columns:
        df['article'] = df['article'].apply(clean_text)
    
    df = df.dropna(subset=['sentence', 'harmfulness_level'])
    
    print(f"Label distribution:")
    print(df['harmfulness_level'].value_counts())
    
    if sample_size is not None and len(df) > sample_size:
        print(f"Sampling {sample_size} rows from {len(df)} for testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return df


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    label_column: str = "harmfulness_level",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Args:
        df: Input DataFrame.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        label_column: Column to use for stratified sampling.
    
    Returns:
        Tuple of (train_df, test_df).
    """
    from sklearn.model_selection import train_test_split
    
    stratify = None
    if label_column in df.columns and len(df[label_column].unique()) > 1:
        class_counts = df[label_column].value_counts()
        min_class_count = class_counts.min()
        if min_class_count >= 2:
            stratify = df[label_column]
        else:
            print(f"Warning: Minimum class count is {min_class_count}. Disabling stratification.")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def pool_annotations_to_soft_labels(
    df: pd.DataFrame,
    text_column: str = "sentence",
    label_column: str = "harmfulness_level",
    label_classes: List[str] = None,
) -> pd.DataFrame:
    """
    Pool multiple annotations per sentence into soft labels (probability vectors).
    
    For sentences with multiple annotations from different annotators, this function
    calculates the proportion of each harmfulness label, creating a soft target
    distribution instead of a single hard label.
    
    Args:
        df: Input DataFrame with potentially multiple rows per sentence.
        text_column: Name of text column to group by.
        label_column: Name of label column.
        label_classes: List of all possible label classes. If None, inferred from data.
    
    Returns:
        DataFrame with one row per unique sentence and soft label columns.
    """
    if label_classes is None:
        label_classes = sorted(df[label_column].unique().tolist())
    
    print(f"Pooling annotations for {len(df)} rows into soft labels...")
    print(f"Label classes: {label_classes}")
    
    grouped = df.groupby(text_column)
    
    pooled_rows = []
    for sentence, group in grouped:
        label_counts = group[label_column].value_counts()
        total = len(group)
        
        soft_labels = {}
        for cls in label_classes:
            soft_labels[f"soft_{cls}"] = label_counts.get(cls, 0) / total
        
        majority_label = label_counts.idxmax()
        
        article = group['article'].iloc[0] if 'article' in group.columns else ""
        
        row = {
            text_column: sentence,
            'article': article,
            label_column: majority_label,
            'n_annotations': total,
            **soft_labels,
        }
        pooled_rows.append(row)
    
    pooled_df = pd.DataFrame(pooled_rows)
    
    print(f"Pooled {len(df)} annotations into {len(pooled_df)} unique sentences")
    print(f"Annotation count distribution:")
    print(pooled_df['n_annotations'].describe())
    
    return pooled_df


def extract_features(
    df: pd.DataFrame, 
    text_column: str = "sentence",
    label_column: str = "harmfulness_level",
) -> Dict[str, Any]:
    """
    Extract text statistics and features for analysis.
    
    Args:
        df: Input DataFrame.
        text_column: Name of text column.
        label_column: Name of label column.
    
    Returns:
        Dictionary with feature statistics.
    """
    stats = {
        "total_samples": len(df),
        "avg_text_length": df[text_column].str.len().mean() if text_column in df.columns else 0,
        "text_length_std": df[text_column].str.len().std() if text_column in df.columns else 0,
    }
    
    if label_column in df.columns:
        stats["label_distribution"] = df[label_column].value_counts().to_dict()
    
    if 'article' in df.columns:
        stats["avg_article_length"] = df['article'].str.len().mean()
    
    for col in ['gender', 'education_level', 'ethnicity', 'nationality']:
        if col in df.columns:
            stats[f"{col}_distribution"] = df[col].value_counts().to_dict()
    
    if 'age' in df.columns:
        stats["age_mean"] = df['age'].mean()
        stats["age_std"] = df['age'].std()
    
    return stats
