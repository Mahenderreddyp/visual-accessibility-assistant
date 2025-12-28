# VizWiz Visual Accessibility Assistant

A fine-tuned vision-language model for assisting visually impaired users with real-time image understanding and answering questions about their visual environment.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Model Export & Quantization](#model-export--quantization)
- [Deployment](#deployment)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🎯 Overview

VizWiz Visual Accessibility Assistant is a production-ready vision-language model fine-tuned on the VizWiz dataset to help visually impaired users understand their visual environment. The model can answer questions about images with high accuracy and appropriately handle unanswerable questions.

**Key Achievements:**
- 🎯 **61.68% accuracy** on VizWiz test set (2.28x improvement over baseline)
- ⚡ **Optimized for edge deployment** via 4-bit quantization (GGUF Q4_K_M)
- 🚀 **Production-ready** with Ollama integration
- 📦 **Compact model** (~1.9 GB deployed size)

---

## 📊 Model Performance

### Training Results

| Metric | Baseline (60 steps) | Production (2500 steps) | Improvement |
|--------|---------------------|-------------------------|-------------|
| **Accuracy** | 27.00% | **61.68%** | **+128%** (2.28x) |
| **Training Time** | ~1 hour | 14.5 hours | - |
| **Final Loss** | 1.2341 | **0.5359** | -56.6% |
| **Test Samples** | 4,319 | 4,319 | - |
| **Correct Predictions** | 1,166 | **2,664** | +1,498 |

### Performance Analysis

**Strengths:**
- ✅ Reduced over-conservative refusals from 94% to ~35%
- ✅ Improved answerable question handling from 5.8% to ~55-60%
- ✅ Maintained good unanswerable detection (~75-80%)
- ✅ Better understanding of VizWiz-specific patterns

**Areas for Improvement:**
- 🔄 Still some over-conservative responses on answerable questions
- 🔄 Spatial reasoning (directional confusion: "bottom" vs "right")
- 🔄 Occasional false positives on unanswerable questions

---

## ✨ Features

- **Visual Question Answering**: Answer questions about images in natural language
- **Unanswerable Detection**: Appropriately identify when questions cannot be answered from the image
- **Production-Ready**: Optimized for deployment with 4-bit quantization
- **Edge-Compatible**: Runs on consumer hardware via Ollama
- **Fast Inference**: Optimized for real-time responses

---

## 🏗️ Architecture

### Base Model
- **Model**: Qwen2.5-VL-3B-Instruct
- **Architecture**: Vision-Language Transformer
- **Parameters**: ~3 billion
- **Vision Encoder**: Pre-trained visual understanding
- **Language Model**: Qwen2.5-3B

### Fine-Tuning Approach
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 16
- **LoRA Alpha**: 16
- **Target Modules**: All linear layers
- **Training Precision**: 4-bit quantization (training), FP16 (inference)

---

## 📚 Dataset

### VizWiz-VQA Dataset

The model is trained on the VizWiz Visual Question Answering dataset, specifically designed for visually impaired users.

**Dataset Statistics:**
- **Total Training Samples**: 20,523
- **Test Samples**: 4,319
- **Question Types**: Real questions from visually impaired users
- **Image Quality**: Variable (realistic, real-world conditions)
- **Unanswerable Questions**: ~30% of dataset

**Dataset Structure:**
```
VizWiz/
├── train/
│   ├── images/          # Training images
│   └── annotations.json # Questions and answers
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

**Sample Questions:**
- "What color is this shirt?"
- "What is the expiration date on this milk?"
- "How much money is this?"
- "What does this label say?"

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 8GB+ GPU VRAM (RTX 3080 or better)
- 32GB+ System RAM
- ~50GB disk space

### Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/visual-accessibility-assistant.git
cd visual-accessibility-assistant

# Create conda environment
conda create -n unsloth_ai python=3.11 -y
conda activate unsloth_ai

# Install PyTorch with CUDA
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

**Core Libraries:**
```
unsloth>=2024.12
transformers>=4.46.3
datasets>=2.19.0
peft>=0.13.2
trl>=0.11.4
accelerate>=0.34.2
bitsandbytes>=0.44.1
```

**Optional (for export):**
```
llama-cpp-python
gguf>=0.1.0
```

---

## 🎓 Training

### Quick Start

```bash
# Download VizWiz dataset
python scripts/01_download_vizwiz.py

# Prepare dataset
python scripts/02_prepare_dataset.py

# Train model
python scripts/03_train_model.py
```

### Training Configuration

**Optimized Settings (RTX 3080 Laptop):**

```python
training_args = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "num_train_epochs": 1,
    "max_steps": 2500,
    "learning_rate": 2e-4,
    "fp16": False,
    "bf16": False,
    "logging_steps": 10,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "models/checkpoints",
    "save_steps": 250,
    "save_total_limit": 3,
}
```

### Training Performance

**Hardware Used:**
- GPU: RTX 3080 Laptop (8GB VRAM)
- CPU: Intel i7-10750H
- RAM: 32GB DDR4
- Storage: NVMe SSD

**Training Metrics:**
- **Speed**: 20.84s/step
- **GPU Utilization**: ~95%
- **GPU Temperature**: 85-88°C (safe range)
- **Total Time**: 14.47 hours
- **Samples/Second**: ~0.19

**Optimization Applied:**
- ✅ Batch size: 2 → 4 (2x GPU utilization)
- ✅ TF32 precision enabled (1.5x speedup)
- ✅ Reduced logging: 1 → 10 steps
- ✅ Reduced checkpointing: 20 → 250 steps
- ✅ Gradient clipping: 1.0
- ✅ Overall speedup: ~2x

---

## 📦 Model Export & Quantization

### Export Pipeline

The model goes through a multi-step export process:

```
Trained Model (LoRA) → Merged FP16 → GGUF F16 → GGUF Q4_K_M
```

### Option 1: Google Colab Export (Recommended)

```python
# 1. Load model in pure FP16
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "models/final/vizwiz_qwen2.5_vl_complete",
    load_in_4bit=False,  # Important: Pure FP16!
    dtype=torch.float16,
)

# 2. Merge LoRA
model = model.merge_and_unload()

# 3. Save
model.save_pretrained("vizwiz_fp16_clean")
tokenizer.save_pretrained("vizwiz_fp16_clean")

# 4. Convert to GGUF F16
!python llama.cpp/convert_hf_to_gguf.py \
    vizwiz_fp16_clean \
    --outfile vizwiz_f16.gguf \
    --outtype f16

# 5. Quantize to Q4_K_M
!./llama.cpp/build/bin/llama-quantize \
    vizwiz_f16.gguf \
    vizwiz_q4_k_m.gguf \
    Q4_K_M
```

### Option 2: Windows Export

```powershell
# 1. Merge and save (Python)
python scripts/05_export_merged.py

# 2. Convert to GGUF
cd llama.cpp
python convert_hf_to_gguf.py ..\models\merged\vizwiz_merged_fp16 `
    --outfile ..\models\gguf\vizwiz_f16.gguf `
    --outtype f16

# 3. Quantize
.\llama-quantize.exe models\gguf\vizwiz_f16.gguf `
    models\gguf\vizwiz_q4_k_m.gguf Q4_K_M
```

### Model Sizes

| Format | Size | Use Case |
|--------|------|----------|
| **Original Training** | 2.3 GB | Training with LoRA |
| **Merged FP16** | 7.1 GB | Full precision inference |
| **GGUF F16** | 5.6 GB | Intermediate format |
| **GGUF Q4_K_M** | **1.9 GB** | **Production deployment** ✅ |

---

## 🚀 Deployment

### Ollama Deployment

#### 1. Install Ollama

```bash
# Windows
winget install Ollama.Ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
```

#### 2. Create Modelfile

Create `models/ollama/Modelfile`:

```dockerfile
FROM ./vizwiz-assistant.gguf

# Context window
PARAMETER num_ctx 4096

# Temperature for responses
PARAMETER temperature 0.7

# Top-p sampling
PARAMETER top_p 0.9

# Chat template (Qwen2-VL format)
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

# Stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# System prompt
SYSTEM """You are a visual assistant for visually impaired users. Analyze images and answer questions clearly and accurately. If you cannot determine the answer from the image, say so honestly."""
```

#### 3. Deploy Model

```bash
# Copy quantized model
copy models\gguf\vizwiz_q4_k_m.gguf models\ollama\vizwiz-assistant.gguf

# Create model in Ollama
cd models\ollama
ollama create vizwiz-assistant-v2 -f Modelfile

# Verify
ollama list
```

#### 4. Test Deployment

```bash
# Text-only test
ollama run vizwiz-assistant-v2 "Hello!"

# With image (if supported by Ollama)
ollama run vizwiz-assistant-v2 "What is in this image?" --image test.jpg
```

---

## 💻 Usage

### Python API

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "models/final/vizwiz_qwen2.5_vl_complete"
)
tokenizer = AutoTokenizer.from_pretrained("models/final/vizwiz_qwen2.5_vl_complete")
processor = AutoProcessor.from_pretrained("models/final/vizwiz_qwen2.5_vl_complete")

# Prepare image and question
image = Image.open("test_image.jpg")
question = "What color is this shirt?"

# Process
inputs = processor(
    text=f"Question: {question}\nAnswer:",
    images=image,
    return_tensors="pt"
)

# Generate answer
outputs = model.generate(**inputs, max_new_tokens=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Q: {question}")
print(f"A: {answer}")
```

### Command Line

```bash
# Using the test script
python test_vision.py --image path/to/image.jpg --question "What is this?"

# Batch evaluation
python evaluate_with_gt.py --model models/final/vizwiz_qwen2.5_vl_complete
```

### Example Questions

```python
# Good questions (model performs well)
"What color is this shirt?"
"What is the expiration date on this milk?"
"How many apples are in the basket?"
"What does this sign say?"

# Unanswerable (model should refuse)
"What is the temperature?" (not visible)
"How does this taste?" (subjective/not visual)
"What time is it in Tokyo?" (requires external knowledge)
```

---

## 📁 Project Structure

```
visual-accessibility-assistant/
├── data/
│   └── vizwiz/                    # VizWiz dataset
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   ├── checkpoints/               # Training checkpoints
│   ├── final/                     # Final trained model
│   │   └── vizwiz_qwen2.5_vl_complete/
│   ├── merged/                    # Merged FP16 models
│   ├── gguf/                      # GGUF format models
│   │   ├── vizwiz_f16.gguf
│   │   └── vizwiz_q4_k_m.gguf
│   └── ollama/                    # Ollama deployment
│       ├── Modelfile
│       └── vizwiz-assistant.gguf
├── scripts/
│   ├── 01_download_vizwiz.py      # Dataset download
│   ├── 02_prepare_dataset.py      # Data preprocessing
│   ├── 03_train_model.py          # Training script
│   ├── 04_evaluate_model.py       # Evaluation
│   ├── 05_export_merged.py        # Model merging
│   └── 06_export_gguf.py          # GGUF conversion
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Dataset analysis
│   └── 02_model_testing.ipynb     # Interactive testing
├── llama.cpp/                     # llama.cpp submodule
├── test_vision.py                 # Testing script
├── evaluate_with_gt.py            # Ground truth evaluation
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── EXPORT_INSTRUCTIONS.md         # Export guide
└── README.md                      # This file
```

---

## 🔧 Technical Details

### Model Architecture Details

```python
Qwen2.5-VL-3B Configuration:
├── Vision Encoder
│   ├── Architecture: Vision Transformer
│   ├── Input: Variable resolution images
│   └── Output: Visual embeddings
├── Language Model (Qwen2.5-3B)
│   ├── Layers: 28
│   ├── Hidden Size: 2048
│   ├── Attention Heads: 16
│   └── Vocabulary: 151,936 tokens
└── LoRA Adapters (Fine-tuned)
    ├── Rank: 16
    ├── Alpha: 16
    ├── Dropout: 0.0
    └── Target: All linear layers
```

### Training Hyperparameters

```yaml
Model:
  base_model: "Qwen/Qwen2.5-VL-3B-Instruct"
  load_in_4bit: true
  max_seq_length: 2048

LoRA:
  r: 16
  lora_alpha: 16
  lora_dropout: 0.0
  target_modules: "all-linear"
  use_rslora: true
  use_gradient_checkpointing: "unsloth"

Training:
  batch_size: 4
  gradient_accumulation: 4
  effective_batch_size: 16
  learning_rate: 2e-4
  optimizer: "adamw_8bit"
  scheduler: "linear"
  warmup_steps: 100
  max_steps: 2500
  weight_decay: 0.01
  gradient_clipping: 1.0

Hardware:
  precision: "FP16"
  device: "CUDA"
  gpu_memory: "8GB"
  mixed_precision: false
```

### Inference Configuration

```python
Generation Parameters:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  do_sample: true
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. GPU Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
```python
# Reduce batch size
per_device_train_batch_size = 2  # Instead of 4

# Enable gradient checkpointing
use_gradient_checkpointing = "unsloth"

# Reduce max sequence length
max_seq_length = 1024  # Instead of 2048
```

#### 2. Quantization Errors

**Error:** `NotImplementedError: Quant method is not yet supported: 'bitsandbytes'`

**Solution:**
```python
# Load model WITHOUT quantization for export
model, tokenizer = FastVisionModel.from_pretrained(
    model_path,
    load_in_4bit=False,  # Important!
    dtype=torch.float16,
)
```

#### 3. llama.cpp Conversion Fails

**Error:** `Failed to detect model architecture`

**Solution:**
```bash
# Use the correct model class
python convert_hf_to_gguf.py model_path \
    --outfile output.gguf \
    --outtype f16 \
    --model-name qwen2_vl  # Specify architecture
```

#### 4. Slow Training

**Solutions:**
- Enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`
- Reduce logging frequency
- Use larger batch sizes if memory allows
- Enable gradient accumulation

### Performance Tips

**Speed Up Training:**
```python
# 1. Enable TF32 (Ampere GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. Increase batch size
per_device_train_batch_size = 4  # or 8 if memory allows

# 3. Reduce checkpointing
save_steps = 500  # Instead of 20

# 4. Use compiled mode (experimental)
torch._dynamo.config.suppress_errors = True
```

**Reduce Memory Usage:**
```python
# 1. Enable gradient checkpointing
use_gradient_checkpointing = True

# 2. Use 8-bit optimizer
optim = "adamw_8bit"

# 3. Reduce sequence length
max_seq_length = 1024

# 4. Enable CPU offloading
device_map = "auto"
```

---

## 🚀 Future Improvements

### Short-term

- [ ] Improve spatial reasoning (directional understanding)
- [ ] Reduce false positives on unanswerable questions
- [ ] Add confidence scores to predictions
- [ ] Implement streaming responses for long answers
- [ ] Add multi-language support

### Medium-term

- [ ] Train on additional accessibility datasets
- [ ] Implement few-shot learning for domain adaptation
- [ ] Add voice input/output integration
- [ ] Create mobile app deployment
- [ ] Implement real-time video understanding

### Long-term

- [ ] Multi-modal reasoning (combine vision + audio)
- [ ] Scene understanding and object detection
- [ ] Personalized model adaptation per user
- [ ] Edge deployment optimization (mobile, IoT)
- [ ] Integration with smart glasses/wearables

---

## 📖 References

### Papers

1. **VizWiz Dataset:**
   - Gurari, D., et al. (2018). "VizWiz Grand Challenge: Answering Visual Questions from Blind People"
   - [Paper Link](https://arxiv.org/abs/1802.08218)

2. **Qwen2-VL:**
   - Bai, J., et al. (2023). "Qwen Technical Report"
   - [Paper Link](https://arxiv.org/abs/2309.16609)

3. **LoRA:**
   - Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
   - [Paper Link](https://arxiv.org/abs/2106.09685)

### Tools & Libraries

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference in C/C++
- [Ollama](https://ollama.com/) - Local LLM deployment
- [Transformers](https://huggingface.co/transformers/) - HuggingFace Transformers
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning

---

## 🙏 Acknowledgments

- **Unsloth Team** for the efficient fine-tuning framework
- **VizWiz Dataset Creators** for the high-quality accessibility dataset
- **Qwen Team** for the excellent base vision-language model
- **llama.cpp Contributors** for the quantization tools
- **Ollama Team** for the deployment platform

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Model Licenses:**
- Base Model (Qwen2.5-VL): Apache 2.0
- VizWiz Dataset: CC BY 4.0

---

## 📧 Contact

**Project Maintainer:** Mahender  
**Email:** your.email@example.com  
**GitHub:** [@yourusername](https://github.com/yourusername)

---

## 🌟 Citation

If you use this work in your research, please cite:

```bibtex
@software{vizwiz_assistant_2024,
  title={VizWiz Visual Accessibility Assistant},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/visual-accessibility-assistant}
}
```

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for accessibility

</div>