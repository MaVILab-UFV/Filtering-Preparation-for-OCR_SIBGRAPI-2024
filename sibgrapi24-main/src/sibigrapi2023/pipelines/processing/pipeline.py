"""
This is a boilerplate pipeline 'processing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    homography,
    homography_segmentation,
    homography_segmentation_batched,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=homography,
            name="homography_func",
            inputs="express_expense",
            outputs=["homography_points", "homography_boxes"],
        ),
        node(
            func=homography_segmentation,
            name="homography_segmentation_func",
            inputs=["masks", "express_expense"],
            outputs=["homography_points_segmentation", "homography_boxes_segmentation"]
        ),
        node(
            func=homography_segmentation_batched,
            name="homography_segmentation_batched_func",
            inputs=["masks_batched", "express_expense"],
            outputs=["homography_points_segmentation_batched_hull", "homography_points_segmentation_batched_hough"]
        ),
    ])
