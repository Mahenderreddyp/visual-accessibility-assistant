# scripts/08_ollama_direct.py
"""
Import model directly to Ollama (skips GGUF conversion)
"""

from pathlib import Path
import shutil

def main():
    print("=" * 70)
    print("🚀 DIRECT OLLAMA IMPORT (Skips GGUF)")
    print("=" * 70)
    
    MERGED_MODEL = "models/merged/vizwiz_merged_fp16"
    OLLAMA_DIR = "models/ollama"
    
    # Create Ollama directory
    Path(OLLAMA_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create Modelfile that points to HF model
    modelfile_content = f"""FROM {Path(MERGED_MODEL).resolve()}

PARAMETER num_ctx 4096
PARAMETER temperature 0.7

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"You are a visual assistant for visually impaired users.\"\"\"
"""
    
    modelfile_path = Path(OLLAMA_DIR) / "Modelfile"
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"\n✅ Created Modelfile: {modelfile_path}")
    
    print("\n" + "=" * 70)
    print("🚀 DEPLOY TO OLLAMA")
    print("=" * 70)
    
    print("\nRun these commands:")
    print(f"   cd {OLLAMA_DIR}")
    print(f"   ollama create vizwiz-assistant-v2 -f Modelfile")
    
    print("\n⚠️  NOTE: Ollama will:")
    print("   1. Read the HuggingFace model")
    print("   2. Automatically quantize it")
    print("   3. Import into Ollama")
    print("   This may take 10-15 minutes")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()