# scripts/simple_gguf_convert.py
"""
Simple GGUF conversion using gguf library
"""

from pathlib import Path
import subprocess
import sys

def main():
    print("=" * 70)
    print("🔄 CONVERTING TO GGUF")
    print("=" * 70)
    
    # First, let's check what we have
    llama_cpp = Path("llama.cpp")
    
    if not llama_cpp.exists():
        print("❌ llama.cpp directory not found!")
        return
    
    # Check for conversion script
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        print("📥 Downloading conversion script...")
        
        import urllib.request
        import os
        
        # Download the main conversion script
        url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
        
        try:
            urllib.request.urlretrieve(url, str(convert_script))
            print("✅ Downloaded convert_hf_to_gguf.py")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nManual steps:")
            print("1. Go to: https://github.com/ggerganov/llama.cpp")
            print("2. Download convert_hf_to_gguf.py")
            print("3. Place it in llama.cpp/ directory")
            return
    
    # Install required packages
    print("\n📦 Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "numpy", "sentencepiece"], check=False)
    
    # Create output directory
    Path("models/gguf").mkdir(parents=True, exist_ok=True)
    
    # Run conversion
    print("\n🔄 Converting to GGUF F16...")
    print("   This may take 3-5 minutes...\n")
    
    cmd = [
        sys.executable,
        str(convert_script),
        "models/merged/vizwiz_merged_fp16",
        "--outfile", "models/gguf/vizwiz_f16.gguf",
        "--outtype", "f16"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        # Check if file was created
        gguf_file = Path("models/gguf/vizwiz_f16.gguf")
        if gguf_file.exists():
            size_gb = gguf_file.stat().st_size / (1024**3)
            print(f"\n✅ Conversion successful!")
            print(f"   File: {gguf_file}")
            print(f"   Size: {size_gb:.2f} GB")
            
            print("\n" + "=" * 70)
            print("✅ STEP 2 COMPLETE!")
            print("=" * 70)
            print("\nNext: Run quantization")
            print("   .\\llama.cpp\\llama-quantize.exe models\\gguf\\vizwiz_f16.gguf `")
            print("       models\\gguf\\vizwiz_q4_k_m.gguf Q4_K_M")
        else:
            print("❌ GGUF file not created")
    else:
        print("❌ Conversion failed")

if __name__ == "__main__":
    main()