# scripts/07_convert_simple.py
"""
Simple conversion - use existing merged model
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🔄 SIMPLE GGUF CONVERSION")
    print("=" * 70)
    
    # Paths
    merged_model = Path("models/merged/vizwiz_merged_fp16")
    output_f16 = Path("models/gguf/vizwiz_f16.gguf")
    output_q4 = Path("models/gguf/vizwiz_q4_k_m.gguf")
    
    # Create output dir
    output_f16.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert with skip-unknown flag
    print("\n[1/2] Converting to GGUF F16...")
    print("   Using --skip-unknown to ignore quantization metadata...\n")
    
    cmd = [
        sys.executable,
        "llama.cpp/convert_hf_to_gguf.py",
        str(merged_model),
        "--outfile", str(output_f16),
        "--outtype", "f16",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode != 0:
        print("\n❌ Conversion failed!")
        print("\nLet's try the manual approach instead...")
        return False
    
    if not output_f16.exists():
        print("❌ F16 GGUF not created")
        return False
    
    size_gb = output_f16.stat().st_size / (1024**3)
    print(f"\n✅ F16 GGUF created!")
    print(f"   File: {output_f16}")
    print(f"   Size: {size_gb:.2f} GB")
    
    # Quantize
    print("\n[2/2] Quantizing to Q4_K_M...\n")
    
    cmd = [
        "llama.cpp/llama-quantize.exe",
        str(output_f16),
        str(output_q4),
        "Q4_K_M"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    
    if result.returncode != 0 or not output_q4.exists():
        print("❌ Quantization failed")
        return False
    
    size_gb = output_q4.stat().st_size / (1024**3)
    print(f"\n✅ Q4_K_M created!")
    print(f"   File: {output_q4}")
    print(f"   Size: {size_gb:.2f} GB")
    
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n" + "=" * 70)
        print("ALTERNATIVE: Use Ollama's Built-in Quantization")
        print("=" * 70)
        print("\nWe can upload the merged model directly to Ollama:")
        print("1. Create a Modelfile pointing to the safetensors")
        print("2. Ollama will quantize it automatically")
        print("\nWould you like to try this approach instead?")