"""Model evaluation nodes for AAVA."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_single_model(
    model_data: Dict[str, Any],
    test_data: pd.DataFrame,
    text_column: str = "sentence",
    label_column: str = "value",
) -> Dict[str, Any]:
    """
    Evaluate a single model on test data.
    
    Args:
        model_data: Dictionary containing the trained model.
        test_data: Test DataFrame.
        text_column: Name of text column.
        label_column: Name of label column.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    model = model_data["model"]
    model_name = model_data["model_name"]
    
    y_true = test_data[label_column].values
    
    y_pred = model.predict(test_data)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    unique_classes = sorted(set(y_true) | set(y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0,
    )
    
    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
        "n_test_samples": len(test_data),
        "classes": [str(c) for c in unique_classes],
    }


def evaluate_all_models(
    trained_models: Dict[str, Any],
    test_data: pd.DataFrame,
    text_column: str = "sentence",
    label_column: str = "value",
) -> Dict[str, Any]:
    """
    Evaluate all trained models and compare performance.
    
    Args:
        trained_models: Dictionary containing all trained models.
        test_data: Test DataFrame.
        text_column: Name of text column.
        label_column: Name of label column.
    
    Returns:
        Dictionary with evaluation results for all models.
    """
    results = {}
    
    for model_name, model_data in trained_models.items():
        print(f"Evaluating {model_name}...")
        results[model_name] = evaluate_single_model(
            model_data, test_data, text_column, label_column
        )
    
    return results


def create_comparison_report(
    evaluation_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a comparison report across all models.
    
    Args:
        evaluation_results: Dictionary with evaluation results for all models.
    
    Returns:
        Comparison report with rankings and summary.
    """
    comparison = []
    
    for model_name, metrics in evaluation_results.items():
        comparison.append({
            "model_name": model_name,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    comparison_df["accuracy_rank"] = comparison_df["accuracy"].rank(ascending=False)
    comparison_df["f1_rank"] = comparison_df["f1_weighted"].rank(ascending=False)
    
    best_accuracy_model = comparison_df.loc[comparison_df["accuracy"].idxmax(), "model_name"]
    best_f1_model = comparison_df.loc[comparison_df["f1_weighted"].idxmax(), "model_name"]
    
    report = {
        "comparison_table": comparison_df.to_dict(orient="records"),
        "best_accuracy_model": best_accuracy_model,
        "best_f1_model": best_f1_model,
        "summary": {
            "total_models_evaluated": len(comparison),
            "accuracy_range": {
                "min": float(comparison_df["accuracy"].min()),
                "max": float(comparison_df["accuracy"].max()),
            },
            "f1_range": {
                "min": float(comparison_df["f1_weighted"].min()),
                "max": float(comparison_df["f1_weighted"].max()),
            },
        },
    }
    
    return report


def select_best_model(
    trained_models: Dict[str, Any],
    comparison_report: Dict[str, Any],
    selection_metric: str = "f1_weighted",
) -> Dict[str, Any]:
    """
    Select the best model based on comparison metrics.
    
    Args:
        trained_models: Dictionary containing all trained models.
        comparison_report: Comparison report from create_comparison_report.
        selection_metric: Metric to use for selection (accuracy or f1_weighted).
    
    Returns:
        The best performing model data.
    """
    if selection_metric == "accuracy":
        best_model_name = comparison_report["best_accuracy_model"]
    else:
        best_model_name = comparison_report["best_f1_model"]
    
    print(f"Selected best model: {best_model_name} (based on {selection_metric})")
    
    return trained_models[best_model_name]


def generate_evaluation_summary(
    comparison_report: Dict[str, Any],
    data_statistics: Dict[str, Any],
) -> str:
    """
    Generate a human-readable evaluation summary.
    
    Args:
        comparison_report: Model comparison report.
        data_statistics: Statistics about the training data.
    
    Returns:
        Formatted summary string.
    """
    lines = [
        "=" * 60,
        "AAVA Model Evaluation Summary",
        "=" * 60,
        "",
        f"Total samples in dataset: {data_statistics.get('total_samples', 'N/A')}",
        f"Average text length: {data_statistics.get('avg_text_length', 0):.0f} characters",
        "",
        "Label Distribution:",
    ]
    
    if "label_distribution" in data_statistics:
        for label, count in data_statistics["label_distribution"].items():
            lines.append(f"  - {label}: {count}")
    
    lines.extend([
        "",
        "-" * 60,
        "Model Performance Comparison",
        "-" * 60,
    ])
    
    for model in comparison_report.get("comparison_table", []):
        lines.extend([
            f"\n{model['model_name'].upper()}:",
            f"  Accuracy:  {model['accuracy']:.4f}",
            f"  F1 Score:  {model['f1_weighted']:.4f}",
            f"  Precision: {model['precision_weighted']:.4f}",
            f"  Recall:    {model['recall_weighted']:.4f}",
        ])
    
    lines.extend([
        "",
        "-" * 60,
        "Recommendations",
        "-" * 60,
        f"Best model by Accuracy: {comparison_report.get('best_accuracy_model', 'N/A')}",
        f"Best model by F1 Score: {comparison_report.get('best_f1_model', 'N/A')}",
        "",
        "=" * 60,
    ])
    
    return "\n".join(lines)
