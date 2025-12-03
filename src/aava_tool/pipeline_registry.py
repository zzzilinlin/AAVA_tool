"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from aava_tool.pipelines import data_processing
from aava_tool.pipelines import model_training
from aava_tool.pipelines import model_evaluation


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()
    model_evaluation_pipeline = model_evaluation.create_pipeline()
    
    return {
        "__default__": data_processing_pipeline + model_training_pipeline + model_evaluation_pipeline,
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,
        "model_evaluation": model_evaluation_pipeline,
        "train": data_processing_pipeline + model_training_pipeline,
        "evaluate": model_evaluation_pipeline,
    }
