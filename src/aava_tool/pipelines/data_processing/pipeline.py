"""Data processing pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import (
    load_and_merge_data,
    preprocess_data,
    create_train_test_split,
    extract_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline."""
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs=[
                    "sentence_data",
                    "raw_annotation_data",
                    "participant_data",
                ],
                outputs="preprocessed_data",
                name="preprocess_data_node",
            ),
            node(
                func=create_train_test_split,
                inputs=["preprocessed_data", "params:test_size", "params:random_state"],
                outputs=["train_data", "test_data"],
                name="train_test_split_node",
            ),
            node(
                func=extract_features,
                inputs="preprocessed_data",
                outputs="data_statistics",
                name="extract_features_node",
            ),
        ]
    )
