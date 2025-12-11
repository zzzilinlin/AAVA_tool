"""Model evaluation pipeline definition."""
from kedro.pipeline import Pipeline, node
from .nodes import (
    evaluate_all_models,
    create_comparison_report,
    select_best_model,
    generate_evaluation_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model evaluation pipeline for harmfulness classification."""
    return Pipeline(
        [
            node(
                func=evaluate_all_models,
                inputs=[
                    "trained_models",
                    "test_data",
                    "params:text_column",
                    "params:label_column",
                ],
                outputs="evaluation_results",
                name="evaluate_all_models_node",
            ),
            node(
                func=create_comparison_report,
                inputs="evaluation_results",
                outputs="comparison_report",
                name="create_comparison_report_node",
            ),
            node(
                func=select_best_model,
                inputs=["trained_models", "comparison_report", "params:selection_metric"],
                outputs="best_model",
                name="select_best_model_node",
            ),
            node(
                func=generate_evaluation_summary,
                inputs=["comparison_report", "data_statistics"],
                outputs="evaluation_summary",
                name="generate_summary_node",
            ),
        ]
    )
