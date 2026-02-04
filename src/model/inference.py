"""
Model loading and inference utilities for CXR-LLaVA.
"""

import torch
from PIL import Image
from typing import Optional
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from peft import PeftModel


class CXRLlavaModel:
    """Wrapper for CXR-LLaVA model inference."""
    
    def __init__(
        self,
        base_model: str = "llava-hf/llava-1.5-7b-hf",
        lora_repo: str = "mohit311/LLaVA-data-json",
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize the CXR-LLaVA model.
        
        Args:
            base_model: HuggingFace model ID for base LLaVA
            lora_repo: HuggingFace repo with LoRA adapter
            device: Device to load model on (auto-detected if None)
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor
        self.processor = LlavaProcessor.from_pretrained(base_model)
        
        # Load base model
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        base = LlavaForConditionalGeneration.from_pretrained(
            base_model,
            **model_kwargs
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            base,
            lora_repo,
            is_trainable=False
        )
        
        self.model.eval()
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
    
    def generate_report(
        self,
        image: Image.Image | str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate a radiology report for the given chest X-ray.
        
        Args:
            image: PIL Image or path to image file
            prompt: Custom prompt (uses default if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated radiology report
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        # Default prompt
        if prompt is None:
            prompt = """<image>
You are an expert radiologist.

Generate a chest X-ray report using EXACTLY this format:

FINDINGS:
- Describe lungs, pleura, heart size, mediastinum, lines or devices if visible.

IMPRESSION:
- Provide a concise clinical summary (1-2 sentences).

Return exactly ONE report."""
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                repetition_penalty=1.1
            )
        
        # Decode
        output = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return self._clean_output(output)
    
    def _clean_output(self, text: str) -> str:
        """Clean and format the model output."""
        # Extract from FINDINGS onwards
        if "FINDINGS:" in text:
            text = text[text.index("FINDINGS:"):]
        
        return text.strip()


def load_model(
    base_model: str = "llava-hf/llava-1.5-7b-hf",
    lora_repo: str = "mohit311/LLaVA-data-json",
    **kwargs
) -> CXRLlavaModel:
    """
    Convenience function to load the CXR-LLaVA model.
    
    Example:
        >>> model = load_model()
        >>> report = model.generate_report("chest_xray.jpg")
        >>> print(report)
    """
    return CXRLlavaModel(base_model, lora_repo, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate report for an image")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--model", default="mohit311/LLaVA-data-json", help="LoRA adapter repo")
    args = parser.parse_args()
    
    model = load_model(lora_repo=args.model)
    report = model.generate_report(args.image)
    
    print("=" * 50)
    print("GENERATED REPORT")
    print("=" * 50)
    print(report)
