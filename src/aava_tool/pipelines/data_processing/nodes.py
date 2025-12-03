"""Data processing nodes for AAVA."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re


def load_and_merge_data(
    raw_data_path: str = None,
    additional_data_path: str = None,
) -> pd.DataFrame:
    """
    Load and merge annotation data from multiple sources.
    
    Args:
        raw_data_path: Path to primary data file (e.g., dynata.csv)
        additional_data_path: Path to additional data file (e.g., progress.csv)
    
    Returns:
        Merged DataFrame with all annotation data.
    """
    dfs = []
    
    if raw_data_path:
        try:
            df = pd.read_csv(raw_data_path)
            dfs.append(df)
        except FileNotFoundError:
            pass
    
    if additional_data_path:
        try:
            df = pd.read_csv(additional_data_path)
            dfs.append(df)
        except FileNotFoundError:
            pass
    
    if not dfs:
        return pd.DataFrame(columns=["text", "category", "justification"])
    
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates()
    
    return merged_df


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"['']", "'", text)
    
    return text


def preprocess_data(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Preprocess the data for model training.
    
    Args:
        df: Input DataFrame with text and labels.
        text_column: Name of the column containing text.
    
    Returns:
        Preprocessed DataFrame.
    """
    df = df.copy()
    
    if text_column in df.columns:
        df[text_column] = df[text_column].apply(clean_text)
        df = df[df[text_column].str.len() > 0]
    
    if "category" in df.columns:
        category_mapping = {
            "Not Harmful": 0,
            "Slightly Harmful": 1,
            "Harmful": 2,
            "Very Harmful": 3,
        }
        df["label"] = df["category"].map(category_mapping)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    
    return df


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Args:
        df: Input DataFrame.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        stratify_column: Column to use for stratified sampling.
    
    Returns:
        Tuple of (train_df, test_df).
    """
    from sklearn.model_selection import train_test_split
    
    stratify = None
    if stratify_column in df.columns and len(df[stratify_column].unique()) > 1:
        class_counts = df[stratify_column].value_counts()
        min_class_count = class_counts.min()
        if min_class_count >= 2:
            stratify = df[stratify_column]
        else:
            print(f"Warning: Minimum class count is {min_class_count}. Disabling stratification.")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def extract_features(df: pd.DataFrame, text_column: str = "text") -> Dict[str, Any]:
    """
    Extract text statistics and features for analysis.
    
    Args:
        df: Input DataFrame.
        text_column: Name of text column.
    
    Returns:
        Dictionary with feature statistics.
    """
    stats = {
        "total_samples": len(df),
        "avg_text_length": df[text_column].str.len().mean() if text_column in df.columns else 0,
        "text_length_std": df[text_column].str.len().std() if text_column in df.columns else 0,
    }
    
    if "label" in df.columns:
        stats["label_distribution"] = df["label"].value_counts().to_dict()
    
    if "category" in df.columns:
        stats["category_distribution"] = df["category"].value_counts().to_dict()
    
    return stats
