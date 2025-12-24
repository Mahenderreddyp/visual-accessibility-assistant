# scripts/06_save_fp16_proper.py
"""
Save model properly in FP16 without any quantization metadata
"""

import os
os.environ["DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from pathlib import Path
import tempfile
import shutil


def main():
    TRAINED_MODEL = "models/final/vizwiz_qwen2.5_vl_complete"
    MERGED_MODEL = "models/merged/vizwiz_merged_fp16"
    
    print("=" * 70)
    print("🔧 SAVE MODEL IN PROPER FP16 (No Quantization)")
    print("=" * 70)
    
    # Remove old merged model if exists
    if Path(MERGED_MODEL).exists():
        print("🗑️  Removing old merged model...")
        shutil.rmtree(MERGED_MODEL)
    
    os.makedirs(MERGED_MODEL, exist_ok=True)
    
    # ========================================
    # Load with transformers (not Unsloth)
    # ========================================
    print("\n[1/2] Loading model with transformers library...")
    print("   Using CPU offloading for 8GB GPU...")
    
    # Create temp offload directory
    offload_dir = tempfile.mkdtemp(prefix="offload_")
    
    try:
        # Load model in FP16 with CPU offloading
        model = AutoModelForVision2Seq.from_pretrained(
            TRAINED_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        
        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL)
        processor = AutoProcessor.from_pretrained(TRAINED_MODEL)
        
        print("✅ Model loaded in FP16")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # Save in proper format
    # ========================================
    print("\n[2/2] Saving as proper FP16 (no quantization)...")
    print("   ⏱️  This may take 5-10 minutes...")
    
    try:
        # Save model
        model.save_pretrained(
            MERGED_MODEL,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        
        # Save tokenizer and processor
        tokenizer.save_pretrained(MERGED_MODEL)
        processor.save_pretrained(MERGED_MODEL)
        
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
    finally:
        # Cleanup
        try:
            shutil.rmtree(offload_dir)
        except:
            pass
    
    # ========================================
    # Now convert to GGUF
    # ========================================
    print("\n" + "=" * 70)
    print("✅ STEP 1 COMPLETE!")
    print("=" * 70)
    
    print("\n📋 NEXT: Convert to GGUF")
    print("\nRun these commands:")
    print("=" * 70)
    print("cd llama.cpp")
    print(f"python convert_hf_to_gguf.py ..\\{MERGED_MODEL} --outfile ..\\models\\gguf\\vizwiz_f16.gguf --outtype f16")
    print("cd ..")
    print()
    print(".\\llama.cpp\\llama-quantize.exe models\\gguf\\vizwiz_f16.gguf models\\gguf\\vizwiz_q4_k_m.gguf Q4_K_M")
    print("=" * 70)


if __name__ == "__main__":
    main()