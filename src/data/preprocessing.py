"""
Data preprocessing utilities for CXR-LLaVA.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Clean and normalize text from radiology reports."""
    text = text.strip()
    text = re.sub(r"GPT:|USER:|ASSISTANT:", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_findings_impression(text: str) -> tuple[str | None, str | None]:
    """Extract FINDINGS and IMPRESSION sections from report text."""
    findings = None
    impression = None
    
    text_upper = text.upper()
    
    if "FINDINGS:" in text_upper:
        start = text_upper.index("FINDINGS:")
        end = text_upper.index("IMPRESSION:") if "IMPRESSION:" in text_upper else len(text)
        findings = text[start:end].replace("FINDINGS:", "").strip()
    
    if "IMPRESSION:" in text_upper:
        start = text_upper.index("IMPRESSION:")
        impression = text[start:].replace("IMPRESSION:", "").strip()
    
    return findings, impression


def clean_dataset(input_path: str, output_path: str) -> int:
    """
    Clean raw dataset and create training-ready format.
    
    Args:
        input_path: Path to raw JSON dataset
        output_path: Path to save cleaned dataset
        
    Returns:
        Number of cleaned samples
    """
    with open(input_path) as f:
        data = json.load(f)
    
    clean_data = []
    
    for ex in data:
        image = ex["image"]
        findings = None
        impression = None
        
        for turn in ex["conversations"]:
            txt = turn["value"]
            
            if "FINDINGS:" in txt.upper():
                findings = txt
            if "IMPRESSION:" in txt.upper():
                impression = txt
        
        if findings and impression:
            findings = clean_text(findings)
            impression = clean_text(impression)
            
            assistant = f"""FINDINGS:
{findings.replace("FINDINGS:", "").strip()}

IMPRESSION:
{impression.replace("IMPRESSION:", "").strip()}
"""
            
            clean_data.append({
                "image": image,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nGenerate a chest X-ray report."
                    },
                    {
                        "from": "gpt",
                        "value": assistant
                    }
                ]
            })
    
    with open(output_path, "w") as f:
        json.dump(clean_data, f, indent=2)
    
    return len(clean_data)


def create_data_collator(processor):
    """Create a data collator for LLaVA training."""
    from PIL import Image
    
    class LlavaDataCollator:
        def __init__(self, processor):
            self.processor = processor
        
        def __call__(self, batch: List[Dict[str, Any]]) -> Dict:
            images = []
            texts = []
            
            for ex in batch:
                images.append(Image.open(ex["image"]).convert("RGB"))
                
                user = ex["conversations"][0]["value"]
                assistant = ex["conversations"][1]["value"]
                texts.append(f"USER: {user}\nASSISTANT: {assistant}")
            
            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
    
    return LlavaDataCollator(processor)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    
    count = clean_dataset(args.input, args.output)
    print(f"Cleaned {count} samples -> {args.output}")
