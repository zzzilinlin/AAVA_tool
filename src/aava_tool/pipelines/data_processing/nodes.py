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


def preprocess_data(
        sentence_data: pd.DataFrame,
        raw_annotation_data: pd.DataFrame,
        participant_data: pd.DataFrame,
        sample_size: int = None,
        ) -> pd.DataFrame:
    """
    Preprocess the data for model training.
    
    Args:
        sentence_data: DataFrame with sentence text.
        raw_annotation_data: DataFrame with annotation results.
        participant_data: DataFrame with participant demographics.
        sample_size: Optional number of rows to sample for testing (None = use all data).
    
    Returns:
        Preprocessed DataFrame.
    """

    # first filter raw_annotation_data to only include responses
    sentence_ids = np.unique(sentence_data['id'])
    raw_annotation_data = raw_annotation_data[raw_annotation_data['unit_id'].isin(sentence_ids)]
     
    # merge coder data into annotation results
    participant_data.set_index('coder', inplace=True)
    raw_annotation_data.set_index('coder', inplace=True)

    merged_annotation_data = pd.merge(
        raw_annotation_data,
        participant_data,
        left_index=True,
        right_index=True,
        how="left"
    )

    # merge sentences into annotation results
    sentence_data.set_index('id', inplace=True)
    merged_annotation_data.reset_index(inplace=True)
    merged_annotation_data.set_index('unit_id', inplace=True)

    merged_annotation_data = pd.merge(
        merged_annotation_data,
        sentence_data,
        left_index=True,
        right_index=True,
        how="left"
    )

    # only keep relevant columns
    relevant_columns = ['value', 'geslacht', 'leeftijd.jaar', 'opleiding', 'politiek_int', 'politiek_pos', 'sentence']
    merged_annotation_data = merged_annotation_data[relevant_columns].dropna()
    
    # filter unrealistic ages
    merged_annotation_data = merged_annotation_data[(merged_annotation_data['leeftijd.jaar'] >= 14) & (merged_annotation_data['leeftijd.jaar'] <= 100)]
    
    result = merged_annotation_data.reset_index(drop=True)
    
    # Sample for testing if sample_size is specified
    if sample_size is not None and len(result) > sample_size:
        print(f"Sampling {sample_size} rows from {len(result)} for testing...")
        result = result.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return result


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: str = "value",
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


def extract_features(df: pd.DataFrame, text_column: str = "sentence") -> Dict[str, Any]:
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
    
    if "value" in df.columns:
        stats["label_distribution"] = df["value"].value_counts().to_dict()
    
    return stats
