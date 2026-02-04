# CXR-LLaVA: Automated Chest X-Ray Report Generation

**Foundation Models Course Project**

*Prasoon Tiwari, Mohit Bagadiya, Kanak Pandit*

---

## Abstract

We present a multimodal AI system that analyzes chest X-ray images and generates structured radiology reports, including **Findings** and **Impression** sections. The system uses a fine-tuned vision-language model (LLaVA-1.5-7B) with LoRA-based parameter-efficient training on real radiology datasets.

**Goal**: To assist radiologists and clinicians by improving reporting efficiency, consistency, and accessibility, without replacing clinical judgment.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Methodology](#methodology)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training Details](#training-details)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Running the Demo](#running-the-demo)
11. [References](#references)

---

## Motivation

| Problem | Our Solution |
|---------|--------------|
| Radiologists face increasing X-ray volumes, leading to time pressure and fatigue-related errors | AI-powered report generation for faster turnaround |
| Manual reporting often lacks consistency across institutions | Standardized structured output format |
| Many regions suffer from shortages of trained radiologists | Scalable, accessible AI assistance |
| Traditional workflows slow down clinical decision-making | Real-time inference with interactive Q&A |

---

## Methodology

Our pipeline consists of four phases:

### Phase 1: Data Structuring & Prompt Generation
- Organized datasets into unified format with image paths and associated reports
- Created instruction-based conversational prompts (image -> report)
- Each sample includes: image path + multi-turn human-GPT style conversations
- Split: 80% Train / 10% Validation / 10% Test

### Phase 2: Model Selection
- **Initial attempt**: Qwen2.5-1.5B - showed hallucination issues and failed to maintain structured format
- **Final choice**: LLaVA-1.5-7B - state-of-the-art open-source multimodal model optimized for visual instruction tuning

### Phase 3: Fine-Tuning with LoRA
- Used parameter-efficient fine-tuning (LoRA) to adapt the model
- Fine-tuned only a small subset of parameters while keeping base model frozen
- Enabled training on limited GPU resources (Kaggle T4 x2)

### Phase 4: Evaluation
- Pathology-level detection metrics (Accuracy, Sensitivity, Specificity, F1)
- Comparison with GPT-4V and Gemini-Pro-Vision
- Clinical evaluation on 500 validation samples

---

## Dataset

We used two publicly available chest X-ray datasets:

| Dataset | Size | Description |
|---------|------|-------------|
| **MIMIC-CXR** | ~377,000 images (4.7 TB) | Large-scale dataset with high-quality paired image-report data from real clinical scenarios |
| **OpenI** | ~7,400 images (75 GB) | Clean annotations, frequently used as benchmark in medical vision-language research |

### Download Dataset

Our processed dataset is available on Kaggle:

**[MIMIC-CXR Processed Dataset](https://www.kaggle.com/datasets/mohitbagadiya/mimic-cxr)**

The dataset is organized into two folders:

```
mimic-cxr/
|-- Main/           # First batch of processed X-ray images
|   |-- p1000/
|   |   |-- i1000.jpg
|   |   +-- ...
|   +-- ...
|
+-- Main2/          # Second batch of processed X-ray images
    |-- p10000/
    |   |-- i10000.jpg
    |   +-- ...
    +-- ...
```

Each patient folder (`p****`) contains the corresponding chest X-ray images (`i****.jpg`).

### Data Format

The training data is structured as JSON with conversational format:

```json
{
  "image": "/path/to/Main/p1000/i1000.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nGenerate a chest X-ray report."},
    {"from": "gpt", "value": "FINDINGS:\n...\n\nIMPRESSION:\n..."}
  ]
}
```

### Data Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 80% | Model fine-tuning |
| Validation | 10% | Hyperparameter tuning |
| Test | 10% | Final evaluation |

---

## Model Architecture

```
                    +------------------+
                    |   Chest X-Ray    |
                    |      Image       |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Vision Encoder  |
                    |   (CLIP/SigLIP)  |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Projection Layer |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   LLaMA/Vicuna   |
                    |    (7B LLM)      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | LoRA Adapters    |
                    | (Fine-tuned)     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Structured Report|
                    | FINDINGS +       |
                    | IMPRESSION       |
                    +------------------+
```

**Why LLaVA-1.5-7B?**
- Connects Vision Encoder (CLIP) with powerful LLM (LLaMA/Vicuna)
- Enables accurate X-ray interpretation and structured report generation
- Well-documented, fits on consumer GPUs, supports interactive Q&A

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `llava-hf/llava-1.5-7b-hf` |
| Fine-tuning Method | LoRA (Low-Rank Adaptation) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | `q_proj`, `v_proj` |
| Batch Size | 1 (per GPU) |
| Gradient Accumulation | 4 (effective batch = 4) |
| Learning Rate | 1e-4 |
| Epochs | 2 |
| Precision | FP16 |
| Hardware | Kaggle T4 x2 GPUs |

---

## Results

### Target Pathologies
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion
- Pneumonia
- Pneumothorax

### Accuracy Comparison

| Pathology | CXR-LLaVA (Ours) | GPT-4V | Gemini-Pro |
|-----------|------------------|--------|------------|
| Cardiomegaly | **0.79** | 0.65 | 0.65 |
| Consolidation | **0.84** | 0.80 | 0.55 |
| Edema | 0.66 | **0.68** | 0.61 |
| Pleural Effusion | **0.72** | 0.66 | 0.53 |
| Pneumonia | **0.83** | 0.67 | 0.71 |
| Pneumothorax | 0.82 | 0.88 | **0.97** |
| **Overall Average** | **0.78** | 0.72 | 0.67 |

### F1 Score Comparison

| Pathology | CXR-LLaVA (Ours) | GPT-4V | Gemini-Pro |
|-----------|------------------|--------|------------|
| Cardiomegaly | **0.86** | 0.77 | 0.78 |
| Consolidation | 0.29 | 0.20 | **0.41** |
| Edema | **0.84** | 0.71 | 0.69 |
| Pleural Effusion | **0.68** | 0.39 | 0.61 |
| Pneumonia | 0.65 | 0.79 | **0.82** |
| Pneumothorax | **0.69** | 0.03 | 0.00 |
| **Overall Average** | **0.66** | 0.48 | 0.55 |

**Key Finding**: Our fine-tuned CXR-LLaVA outperforms both GPT-4V and Gemini-Pro on average accuracy (0.78 vs 0.72/0.67) and F1 score (0.66 vs 0.48/0.55).

---

## Project Structure

```
CXR-LLaVA/
|-- README.md                 # This file
|-- LICENSE                   # MIT License
|-- requirements.txt          # Python dependencies
|-- .gitignore
|
|-- notebooks/
|   +-- llava_finetuning.ipynb    # Training notebook (Kaggle)
|
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   +-- preprocessing.py      # Data cleaning & collator
|   |-- model/
|   |   |-- __init__.py
|   |   +-- inference.py          # Model loading & inference
|   +-- evaluation/
|       |-- __init__.py
|       +-- metrics.py            # BLEU, ROUGE, pathology F1
|
|-- configs/
|   +-- training_config.yaml      # Training hyperparameters
|
|-- scripts/
|   |-- train.py                  # Training script
|   +-- evaluate.py               # Evaluation script
|
|-- demo/
|   +-- gradio_app.py             # Web interface
|
+-- models/
    +-- lora-adapter/             # Downloaded LoRA weights
        |-- adapter_model.safetensors
        +-- adapter_config.json
```

---

## Installation

### Hardware Requirements

| Component | Inference (Demo) | Training |
|-----------|------------------|----------|
| **GPU** | NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, RTX 4090, A100) | 2x NVIDIA T4/V100 or 1x A100 |
| **GPU VRAM** | 16GB minimum | 32GB+ recommended |
| **RAM** | 16GB minimum | 32GB+ recommended |
| **Storage** | ~50GB (model weights + dependencies) | 100GB+ (including dataset) |
| **CUDA** | CUDA 11.8+ | CUDA 11.8+ |

**Note**: The base LLaVA model (~14GB) downloads automatically on first run. For CPU-only inference, expect significantly slower performance (not recommended).

**Tested Environments**:
- Kaggle Notebooks (T4 x2 GPUs) - Used for training
- Google Colab Pro (A100) - Works for inference
- Local RTX 3090/4090 - Works for inference

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (see requirements above)
- ~50GB disk space for model weights

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CXR-LLaVA.git
cd CXR-LLaVA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Model (if not included)

The LoRA adapter is available on HuggingFace. To download:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mohit311/LLaVA-data-json",
    local_dir="models/lora-adapter"
)
```

---

## Usage

### Quick Inference

```python
from src.model.inference import load_model

# Load model (uses local adapter if available)
model = load_model(
    lora_repo="models/lora-adapter"  # Local path
    # or: lora_repo="mohit311/LLaVA-data-json"  # From HuggingFace
)

# Generate report
report = model.generate_report("path/to/chest_xray.jpg")
print(report)
```

### Training

```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --data data/train_clean.json \
    --hf_token YOUR_HF_TOKEN
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model_path models/lora-adapter \
    --data_path data/val.json \
    --num_samples 500 \
    --compare_baseline
```

---

## Running the Demo

### Option 1: Using Local Model (Recommended)

```bash
# Run with local model folder
python demo/gradio_app.py --local_model models/lora-adapter
```

### Option 2: Using HuggingFace Model

```bash
# Run with HuggingFace model (downloads automatically)
python demo/gradio_app.py
```

### Option 3: With Public Sharing

```bash
# Creates a public URL for sharing
python demo/gradio_app.py --local_model models/lora-adapter --share
```

Once running, open the URL shown in terminal (typically `http://127.0.0.1:7860`).

**Demo Features**:
1. Upload a chest X-ray image
2. Enter a question (or use the default report generation prompt)
3. Click "Analyze" to generate the report
4. View structured FINDINGS and IMPRESSION sections

---

## Example Output

**Input**: Chest X-ray image + "Generate a chest X-ray report."

**Generated Report**:
```
FINDINGS:
The lungs are clear bilaterally. No focal consolidation, pleural effusion,
or pneumothorax is seen. The cardiac silhouette is within normal limits.
The mediastinal contours are unremarkable. No acute osseous abnormality.

IMPRESSION:
No acute cardiopulmonary abnormality.
```

---

## Code Explanation

### Key Components

| File | Purpose |
|------|---------|
| `src/data/preprocessing.py` | Cleans raw reports, extracts FINDINGS/IMPRESSION, creates training-ready JSON |
| `src/model/inference.py` | `CXRLlavaModel` class that loads base model + LoRA adapter and runs inference |
| `src/evaluation/metrics.py` | Computes pathology detection (F1, sensitivity, specificity) and NLG metrics (BLEU, ROUGE) |
| `demo/gradio_app.py` | Interactive web interface using Gradio |
| `scripts/train.py` | Full training pipeline with config loading |
| `scripts/evaluate.py` | Evaluation pipeline comparing baseline vs fine-tuned |

### How Inference Works

1. **Image Processing**: CLIP vision encoder converts X-ray to visual embeddings
2. **Prompt Engineering**: Image token `<image>` + text prompt are tokenized together
3. **Generation**: LLaVA generates tokens autoregressively
4. **Post-processing**: Extract FINDINGS and IMPRESSION sections from output

---

## Future Work

- **Larger-scale training**: Full MIMIC-CXR dataset (~377K images)
- **Stronger evaluation**: CheXbert, RadGraph clinical NLP metrics
- **Multi-view support**: Frontal + lateral X-rays, CT slices
- **Clinical alignment**: RLHF to reduce hallucinations
- **Deployment**: Production-ready web application

---

## References

1. Liu et al., "Visual Instruction Tuning" (LLaVA), NeurIPS 2023
2. Radford et al., "CLIP: Learning Transferable Visual Models", ICML 2021
3. Boecking et al., "BioViL: Self-Supervised Vision-Language Pretraining for Radiology", ICCV 2022
4. [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning

---

## Disclaimer

This model is intended for **research and educational purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
