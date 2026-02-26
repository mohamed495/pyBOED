"""Convenience re-exports for randomized numerical linear algebra."""

from boed.algebra.rnla_core import (
    RNLA,
    rangefinder,
    randomized_svd,
    assessment,
    demo_plot,
    demo_image,
    low_rank_approx,
)

__all__ = [
    "RNLA",
    "rangefinder",
    "randomized_svd",
    "assessment",
    "demo_plot",
    "demo_image",
    "low_rank_approx",
]
