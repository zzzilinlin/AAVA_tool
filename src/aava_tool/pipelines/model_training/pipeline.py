"""Model training pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import train_all_models


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline."""
    return Pipeline(
        [
            node(
                func=train_all_models,
                inputs=[
                    "train_data",
                    "params:tfidf",
                    "params:logistic_regression",
                    "params:sbert",
                    "params:ridge",
                ],
                outputs="trained_models",
                name="train_all_models_node",
            ),
        ]
    )
