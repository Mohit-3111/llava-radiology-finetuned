"""
Evaluation metrics for CXR-LLaVA radiology report generation.
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# Default pathologies to detect
PATHOLOGIES = [
    "cardiomegaly",
    "consolidation", 
    "edema",
    "pleural effusion",
    "pneumonia",
    "pneumothorax",
    "atelectasis",
    "lung opacity",
]


def extract_pathologies(text: str, pathologies: List[str] = None) -> set:
    """
    Extract mentioned pathologies from report text.
    
    Args:
        text: Report text to analyze
        pathologies: List of pathologies to detect (uses defaults if None)
        
    Returns:
        Set of detected pathology names
    """
    if pathologies is None:
        pathologies = PATHOLOGIES
    
    text = text.lower()
    found = set()
    
    for p in pathologies:
        if re.search(rf"\b{re.escape(p)}\b", text):
            found.add(p)
    
    return found


def compute_pathology_metrics(
    predictions: List[str],
    references: List[str],
    pathologies: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute pathology-level detection metrics.
    
    Args:
        predictions: List of predicted reports
        references: List of ground truth reports
        pathologies: List of pathologies to evaluate
        
    Returns:
        Dictionary with metrics per pathology
    """
    if pathologies is None:
        pathologies = PATHOLOGIES
    
    results = {}
    
    for pathology in pathologies:
        y_true = []
        y_pred = []
        
        for pred, ref in zip(predictions, references):
            pred_labels = extract_pathologies(pred, [pathology])
            ref_labels = extract_pathologies(ref, [pathology])
            
            y_true.append(int(pathology in ref_labels))
            y_pred.append(int(pathology in pred_labels))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Compute metrics
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
        
        results[pathology] = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    
    return results


def compute_aggregate_metrics(
    predictions: List[str],
    references: List[str],
    pathologies: List[str] = None
) -> Dict[str, float]:
    """
    Compute aggregate pathology detection metrics.
    
    Args:
        predictions: List of predicted reports
        references: List of ground truth reports
        pathologies: List of pathologies to evaluate
        
    Returns:
        Dictionary with aggregate precision, recall, and F1
    """
    if pathologies is None:
        pathologies = PATHOLOGIES
    
    y_true = []
    y_pred = []
    
    for pred, ref in zip(predictions, references):
        pred_labels = extract_pathologies(pred, pathologies)
        ref_labels = extract_pathologies(ref, pathologies)
        
        for p in pathologies:
            y_true.append(int(p in ref_labels))
            y_pred.append(int(p in pred_labels))
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_nlg_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute NLG metrics (BLEU, ROUGE) for report generation.
    
    Args:
        predictions: List of predicted reports
        references: List of ground truth reports
        
    Returns:
        Dictionary with BLEU and ROUGE scores
    """
    import evaluate
    
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    
    # BLEU expects list of references per prediction
    refs_for_bleu = [[r] for r in references]
    
    bleu_score = bleu.compute(predictions=predictions, references=refs_for_bleu)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    
    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
    }


def format_results_table(
    pathology_metrics: Dict[str, Dict[str, float]],
    nlg_metrics: Dict[str, float] = None
) -> str:
    """
    Format evaluation results as a markdown table.
    
    Args:
        pathology_metrics: Per-pathology metrics
        nlg_metrics: Optional NLG metrics
        
    Returns:
        Formatted markdown table string
    """
    lines = []
    
    # Pathology table
    lines.append("## Pathology Detection Results\n")
    lines.append("| Pathology | Accuracy | Sensitivity | Specificity | F1 |")
    lines.append("|-----------|----------|-------------|-------------|-----|")
    
    for pathology, metrics in pathology_metrics.items():
        lines.append(
            f"| {pathology.title()} | "
            f"{metrics['accuracy']:.2f} | "
            f"{metrics['sensitivity']:.2f} | "
            f"{metrics['specificity']:.2f} | "
            f"{metrics['f1']:.2f} |"
        )
    
    if nlg_metrics:
        lines.append("\n## NLG Metrics\n")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric, score in nlg_metrics.items():
            lines.append(f"| {metric.upper()} | {score:.4f} |")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    preds = [
        "FINDINGS: Cardiomegaly is present. IMPRESSION: Heart is enlarged.",
        "FINDINGS: Lungs clear. IMPRESSION: Normal chest X-ray."
    ]
    refs = [
        "FINDINGS: The heart is enlarged. IMPRESSION: Cardiomegaly.",
        "FINDINGS: Clear lungs. IMPRESSION: No acute findings."
    ]
    
    pathology_results = compute_pathology_metrics(preds, refs)
    aggregate_results = compute_aggregate_metrics(preds, refs)
    
    print("Pathology Results:")
    for p, m in pathology_results.items():
        print(f"  {p}: F1={m['f1']:.2f}")
    
    print(f"\nAggregate F1: {aggregate_results['f1']:.2f}")
