"""
This is a boilerplate pipeline 'image_perturbation'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    generate_ocr_perturbation_results,
    generate_ocr_perturbation_report,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
      node(
        func=generate_ocr_perturbation_results,
        name="generate_ocr_perturbation_results_func",
        inputs=["express_test", "labels"],
        outputs="image_perturbation_metrics",
      ),
      node(
        func=generate_ocr_perturbation_report,
        name="generate_ocr_perturbation_report_func",
        inputs="image_perturbation_metrics",
        outputs="image_perturbation_report",
      ),
    ])
