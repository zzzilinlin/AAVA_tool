"""Data processing pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data, create_train_test_split, extract_features


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline for harmfulness classification."""
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["dummy_data", "params:sample_size"],
                outputs="preprocessed_data",
                name="preprocess_data_node",
            ),
            node(
                func=extract_features,
                inputs=["preprocessed_data", "params:text_column", "params:label_column"],
                outputs="data_statistics",
                name="extract_features_node",
            ),
            node(
                func=create_train_test_split,
                inputs=[
                    "preprocessed_data",
                    "params:test_size",
                    "params:random_state",
                    "params:label_column",
                ],
                outputs=["train_data", "test_data"],
                name="train_test_split_node",
            ),
        ]
    )
