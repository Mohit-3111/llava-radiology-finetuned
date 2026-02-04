"""
Gradio web interface for CXR-LLaVA medical image diagnosis.

Usage:
    python demo/gradio_app.py --local_model models/lora-adapter
    python demo/gradio_app.py --share
"""

import argparse
import gradio as gr
import torch
import gc
from PIL import Image
from pathlib import Path

# Global model cache
_model = None
_processor = None
_device = None
_model_path = None


def load_model_once(model_path: str = None):
    """Load model once on startup."""
    global _model, _processor, _device, _model_path
    
    if _model is not None:
        return
    
    from transformers import LlavaForConditionalGeneration, LlavaProcessor
    from peft import PeftModel
    
    BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
    
    # Use local model path if provided, otherwise use HuggingFace
    if model_path:
        _model_path = model_path
        print(f"Using local model: {model_path}")
    else:
        _model_path = "mohit311/LLaVA-data-json"
        print(f"Using HuggingFace model: {_model_path}")
    
    print("Loading processor...")
    _processor = LlavaProcessor.from_pretrained(BASE_MODEL)
    
    print("Loading base model...")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32
    
    base_model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if _device == "cuda" else None
    )
    
    print(f"Loading LoRA adapter from: {_model_path}")
    _model = PeftModel.from_pretrained(
        base_model,
        _model_path,
        is_trainable=False
    ).eval()
    
    print(f"Model loaded on {_device.upper()}!")


def diagnose_image(image: Image.Image, question: str) -> str:
    """Generate diagnosis from image and question."""
    
    if image is None:
        return "Please upload an image first!"
    
    if not question or question.strip() == "":
        return "Please enter a question!"
    
    try:
        # Ensure model is loaded
        load_model_once()
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Create prompt
        prompt = f"<image>\n{question}"
        
        # Process inputs
        inputs = _processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1
            )
        
        # Decode output (skip input tokens)
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, input_length:]
        response = _processor.decode(new_tokens, skip_special_tokens=True)
        
        # Clean up
        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return response.strip()
    
    except Exception as e:
        return f"Error: {str(e)}"


def create_demo() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="CXR-LLaVA Medical Diagnosis",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # CXR-LLaVA: Automated Chest X-Ray Report Generation
        
        Upload a chest X-ray image and ask questions about it. The model will generate
        structured radiology reports with FINDINGS and IMPRESSION sections.
        
        **Model**: LLaVA 1.5 7B fine-tuned on MIMIC-CXR and OpenI datasets.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Image")
                image_input = gr.Image(label="Chest X-Ray", type="pil")
                
                gr.Markdown("### Ask Your Question")
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Generate a chest X-ray report with FINDINGS and IMPRESSION.",
                    lines=3,
                    value="Generate a detailed chest X-ray report with FINDINGS and IMPRESSION sections."
                )
                
                submit_button = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column():
                gr.Markdown("### AI Response")
                output_text = gr.Textbox(
                    label="Generated Report",
                    lines=15,
                    interactive=False
                )
        
        # Example questions
        gr.Markdown("### Example Questions")
        
        example_questions = [
            "Generate a chest X-ray report with FINDINGS and IMPRESSION sections.",
            "What abnormalities can you identify in this X-ray?",
            "Is there any evidence of cardiomegaly or pleural effusion?",
            "Describe the key findings in this chest X-ray.",
            "Are there any signs of pneumonia or consolidation?"
        ]
        
        gr.Examples(
            examples=[[None, q] for q in example_questions],
            inputs=[image_input, question_input],
            label="Try these questions (add your own image)"
        )
        
        # Connect button
        submit_button.click(
            fn=diagnose_image,
            inputs=[image_input, question_input],
            outputs=output_text
        )
        
        gr.Markdown("""
        ---
        ### Technical Details
        - **Base Model**: LLaVA 1.5 7B (llava-hf/llava-1.5-7b-hf)
        - **Fine-tuning**: LoRA on MIMIC-CXR + OpenI datasets
        - **Target Pathologies**: Cardiomegaly, Consolidation, Edema, Pleural Effusion, Pneumonia, Pneumothorax
        
        ### How It Works
        1. The Vision Encoder (CLIP) processes the X-ray image into visual embeddings
        2. These embeddings are projected and combined with the text prompt
        3. The LLaMA-based LLM generates a structured radiology report
        4. Post-processing extracts FINDINGS and IMPRESSION sections
        
        **Disclaimer**: This tool is for educational and research purposes only.
        It should not be used as a substitute for professional medical diagnosis.
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="CXR-LLaVA Demo")
    parser.add_argument(
        "--local_model",
        type=str,
        default=None,
        help="Path to local LoRA adapter folder (e.g., models/lora-adapter)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    args = parser.parse_args()
    
    # Validate local model path if provided
    if args.local_model:
        model_path = Path(args.local_model)
        if not model_path.exists():
            print(f"Warning: Local model path does not exist: {args.local_model}")
            print("Will attempt to download from HuggingFace instead.")
            args.local_model = None
        elif not (model_path / "adapter_config.json").exists():
            print(f"Warning: No adapter_config.json found in {args.local_model}")
            print("Will attempt to download from HuggingFace instead.")
            args.local_model = None
    
    # Pre-load model
    print("Starting CXR-LLaVA Demo...")
    load_model_once(args.local_model)
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=args.share,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
