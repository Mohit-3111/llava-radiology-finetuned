"""CXR-LLaVA: Fine-tuned vision-language model for chest X-ray report generation."""

from .data.preprocessing import clean_dataset, create_data_collator
from .model.inference import CXRLlavaModel, load_model
from .evaluation.metrics import (
    compute_pathology_metrics,
    compute_aggregate_metrics,
    compute_nlg_metrics,
)

__version__ = "1.0.0"
__author__ = "Mohit"

__all__ = [
    "clean_dataset",
    "create_data_collator",
    "CXRLlavaModel",
    "load_model",
    "compute_pathology_metrics",
    "compute_aggregate_metrics",
    "compute_nlg_metrics",
]
