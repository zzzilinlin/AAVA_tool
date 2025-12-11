"""Model training pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import train_all_models


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline for harmfulness classification."""
    return Pipeline(
        [
            node(
                func=train_all_models,
                inputs=[
                    "train_data",
                    "params:tfidf",
                    "params:logistic_regression",
                    "params:sbert",
                    "params:text_column",
                    "params:label_column",
                    "params:cat_columns",
                    "params:num_columns",
                ],
                outputs="trained_models",
                name="train_all_models_node",
            ),
        ]
    )
