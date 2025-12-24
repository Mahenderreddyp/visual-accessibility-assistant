# scripts/05_manual_export.py
"""
Manual Export for Windows
1. Save model as HuggingFace format
2. Manual conversion instructions
"""

import os
os.environ["DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from unsloth import FastVisionModel
from pathlib import Path


def main():
    # ========================================
    # Configuration
    # ========================================
    TRAINED_MODEL = "models/final/vizwiz_qwen2.5_vl_complete"
    MERGED_MODEL = "models/merged/vizwiz_merged_fp16"
    
    print("=" * 70)
    print("🔧 MANUAL EXPORT PIPELINE (Windows Compatible)")
    print("=" * 70)
    print(f"Source: {TRAINED_MODEL}")
    print(f"Target: {MERGED_MODEL}")
    print("=" * 70)
    
    os.makedirs("models/merged", exist_ok=True)
    
    # ========================================
    # Step 1: Load Model in 4-bit
    # ========================================
    print("\n[1/3] Loading model in 4-bit (memory efficient)...")
    
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            TRAINED_MODEL,
            load_in_4bit=True,
            max_seq_length=2048,
        )
        
        print("✅ Model loaded in 4-bit")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return
    
    # ========================================
    # Step 2: Merge LoRA
    # ========================================
    print("\n[2/3] Merging LoRA adapters...")
    
    try:
        model = model.merge_and_unload()
        print("✅ LoRA merged")
    except Exception as e:
        print(f"❌ Failed to merge: {e}")
        return
    
    # ========================================
    # Step 3: Save as Standard HF Format
    # ========================================
    print("\n[3/3] Saving merged model (FP16 HuggingFace format)...")
    print("   ⏱️  This may take 5-10 minutes...")
    
    try:
        # Save model
        model.save_pretrained(
            MERGED_MODEL,
            safe_serialization=True,
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(MERGED_MODEL)
        
        print(f"✅ Model saved to: {MERGED_MODEL}")
        
        # Check size
        total_size = 0
        for file in Path(MERGED_MODEL).rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        
        size_gb = total_size / (1024**3)
        print(f"   Size: {size_gb:.2f} GB")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # Next Steps
    # ========================================
    print("\n" + "=" * 70)
    print("✅ STEP 1 COMPLETE - Model Saved!")
    print("=" * 70)
    
    print("\n📋 MANUAL CONVERSION STEPS:")
    
    print("\n2️⃣  Convert to GGUF F16 (run in PowerShell):")
    print("=" * 70)
    print("cd llama.cpp")
    print("python convert_hf_to_gguf.py ..\\models\\merged\\vizwiz_merged_fp16 `")
    print("    --outfile ..\\models\\gguf\\vizwiz_f16.gguf `")
    print("    --outtype f16")
    print("cd ..")
    
    print("\n3️⃣  Quantize to Q4_K_M (run in PowerShell):")
    print("=" * 70)
    print("mkdir models\\gguf -Force")
    print(".\\llama.cpp\\llama-quantize.exe models\\gguf\\vizwiz_f16.gguf `")
    print("    models\\gguf\\vizwiz_q4_k_m.gguf Q4_K_M")
    
    print("\n4️⃣  Prepare for Ollama (run in PowerShell):")
    print("=" * 70)
    print("mkdir models\\ollama -Force")
    print("copy models\\gguf\\vizwiz_q4_k_m.gguf models\\ollama\\vizwiz-assistant.gguf")
    
    print("\n5️⃣  Create Modelfile:")
    print("=" * 70)
    print('$modelfile = @"')
    print('FROM ./vizwiz-assistant.gguf')
    print('')
    print('PARAMETER num_ctx 4096')
    print('PARAMETER temperature 0.7')
    print('')
    print('TEMPLATE """{{ if .System }}<|im_start|>system')
    print('{{ .System }}<|im_end|>')
    print('{{ end }}{{ if .Prompt }}<|im_start|>user')
    print('{{ .Prompt }}<|im_end|>')
    print('{{ end }}<|im_start|>assistant')
    print('"""')
    print('')
    print('PARAMETER stop "<|im_start|>"')
    print('PARAMETER stop "<|im_end|>"')
    print('')
    print('SYSTEM """You are a visual assistant for visually impaired users."""')
    print('"@')
    print('')
    print('$modelfile | Out-File -FilePath models\\ollama\\Modelfile -Encoding utf8')
    
    print("\n6️⃣  Deploy to Ollama:")
    print("=" * 70)
    print("cd models\\ollama")
    print("ollama create vizwiz-assistant-v2 -f Modelfile")
    
    print("\n" + "=" * 70)
    print("💾 TIP: Copy the commands above and run them one by one!")
    print("=" * 70)
    
    # Save instructions to file
    save_instructions()


def save_instructions():
    """Save step-by-step instructions"""
    
    instructions = """# VizWiz Model Export Instructions (Windows)

## Step 1: Model Saved ✅
The merged model has been saved to: models/merged/vizwiz_merged_fp16

## Step 2: Convert to GGUF F16

Open PowerShell and run:
```powershell
cd llama.cpp
python convert_hf_to_gguf.py ..\\models\\merged\\vizwiz_merged_fp16 `
    --outfile ..\\models\\gguf\\vizwiz_f16.gguf `
    --outtype f16
cd ..
```

Expected: Creates vizwiz_f16.gguf (~5.6 GB)

## Step 3: Quantize to Q4_K_M
```powershell
mkdir models\\gguf -Force
.\\llama.cpp\\llama-quantize.exe models\\gguf\\vizwiz_f16.gguf `
    models\\gguf\\vizwiz_q4_k_m.gguf Q4_K_M
```

Expected: Creates vizwiz_q4_k_m.gguf (~1.9 GB)

## Step 4: Prepare for Ollama
```powershell
mkdir models\\ollama -Force
copy models\\gguf\\vizwiz_q4_k_m.gguf models\\ollama\\vizwiz-assistant.gguf
```

## Step 5: Create Modelfile
```powershell
$modelfile = @"
FROM ./vizwiz-assistant.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0.7

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"You are a visual assistant for visually impaired users.\"\"\"
"@

$modelfile | Out-File -FilePath models\\ollama\\Modelfile -Encoding utf8
```

## Step 6: Deploy to Ollama
```powershell
cd models\\ollama
ollama create vizwiz-assistant-v2 -f Modelfile
```

## Step 7: Test
```powershell
ollama list
ollama run vizwiz-assistant-v2 "Hello!"
```

## Troubleshooting

- **Convert fails**: Make sure you're in llama.cpp directory
- **Quantize fails**: Check that vizwiz_f16.gguf exists
- **Ollama create fails**: Verify vizwiz-assistant.gguf exists in models/ollama

## Expected Timeline

- Step 2 (Convert): 3-5 minutes
- Step 3 (Quantize): 5-10 minutes  
- Step 4-6: < 1 minute each
- Total: ~15-20 minutes
"""
    
    with open("EXPORT_INSTRUCTIONS.md", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"\n📄 Instructions saved to: EXPORT_INSTRUCTIONS.md")


if __name__ == "__main__":
    main()