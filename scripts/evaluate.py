"""
Evaluation script for CXR-LLaVA model.
"""

import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from peft import PeftModel

import sys
sys.path.append(".")
from src.evaluation.metrics import (
    compute_pathology_metrics,
    compute_aggregate_metrics,
    compute_nlg_metrics,
    format_results_table,
    PATHOLOGIES
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CXR-LLaVA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mohit311/LLaVA-data-json",
        help="Path or HF repo ID for LoRA adapter"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Base model ID"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data JSON"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Also evaluate baseline model for comparison"
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load processor
    print("Loading processor...")
    processor = LlavaProcessor.from_pretrained(args.base_model)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path)["train"]
    
    if args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
    
    print(f"Evaluating on {len(dataset)} samples")
    
    def run_evaluation(model, name):
        """Run evaluation on a model."""
        predictions = []
        references = []
        
        for sample in tqdm(dataset, desc=f"Evaluating {name}"):
            image = Image.open(sample["image"]).convert("RGB")
            prompt = sample["conversations"][0]["value"]
            gt = sample["conversations"][1]["value"]
            
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )
            
            pred = processor.decode(output_ids[0], skip_special_tokens=True)
            
            predictions.append(pred)
            references.append(gt)
        
        # Compute metrics
        pathology_metrics = compute_pathology_metrics(predictions, references)
        aggregate_metrics = compute_aggregate_metrics(predictions, references)
        
        try:
            nlg_metrics = compute_nlg_metrics(predictions, references)
        except Exception as e:
            print(f"Warning: Could not compute NLG metrics: {e}")
            nlg_metrics = {}
        
        return {
            "pathology": pathology_metrics,
            "aggregate": aggregate_metrics,
            "nlg": nlg_metrics,
            "predictions": predictions,
            "references": references
        }
    
    results = {}
    
    # Evaluate baseline if requested
    if args.compare_baseline:
        print("\nLoading baseline model...")
        baseline_model = LlavaForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
        
        results["baseline"] = run_evaluation(baseline_model, "Baseline")
        
        del baseline_model
        torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    print("\nLoading fine-tuned model...")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    finetuned_model = PeftModel.from_pretrained(
        base_model,
        args.model_path
    ).eval()
    
    results["finetuned"] = run_evaluation(finetuned_model, "Fine-tuned")
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    if "baseline" in results:
        print("\n--- Baseline ---")
        print(f"Aggregate F1: {results['baseline']['aggregate']['f1']:.4f}")
        if results['baseline']['nlg']:
            print(f"BLEU: {results['baseline']['nlg'].get('bleu', 'N/A')}")
            print(f"ROUGE-L: {results['baseline']['nlg'].get('rougeL', 'N/A')}")
    
    print("\n--- Fine-tuned ---")
    print(f"Aggregate F1: {results['finetuned']['aggregate']['f1']:.4f}")
    if results['finetuned']['nlg']:
        print(f"BLEU: {results['finetuned']['nlg'].get('bleu', 'N/A')}")
        print(f"ROUGE-L: {results['finetuned']['nlg'].get('rougeL', 'N/A')}")
    
    print("\n--- Pathology Results ---")
    print(format_results_table(
        results["finetuned"]["pathology"],
        results["finetuned"]["nlg"]
    ))
    
    # Save results (without predictions/references to keep file small)
    output_results = {
        k: {
            "pathology": v["pathology"],
            "aggregate": v["aggregate"],
            "nlg": v["nlg"]
        }
        for k, v in results.items()
    }
    
    with open(args.output, "w") as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
