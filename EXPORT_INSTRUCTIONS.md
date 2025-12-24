# VizWiz Model Export Instructions (Windows)

## Step 1: Model Saved ✅
The merged model has been saved to: models/merged/vizwiz_merged_fp16

## Step 2: Convert to GGUF F16

Open PowerShell and run:
```powershell
cd llama.cpp
python convert_hf_to_gguf.py ..\models\merged\vizwiz_merged_fp16 `
    --outfile ..\models\gguf\vizwiz_f16.gguf `
    --outtype f16
cd ..
```

Expected: Creates vizwiz_f16.gguf (~5.6 GB)

## Step 3: Quantize to Q4_K_M
```powershell
mkdir models\gguf -Force
.\llama.cpp\llama-quantize.exe models\gguf\vizwiz_f16.gguf `
    models\gguf\vizwiz_q4_k_m.gguf Q4_K_M
```

Expected: Creates vizwiz_q4_k_m.gguf (~1.9 GB)

## Step 4: Prepare for Ollama
```powershell
mkdir models\ollama -Force
copy models\gguf\vizwiz_q4_k_m.gguf models\ollama\vizwiz-assistant.gguf
```

## Step 5: Create Modelfile
```powershell
$modelfile = @"
FROM ./vizwiz-assistant.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0.7

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

SYSTEM """You are a visual assistant for visually impaired users."""
"@

$modelfile | Out-File -FilePath models\ollama\Modelfile -Encoding utf8
```

## Step 6: Deploy to Ollama
```powershell
cd models\ollama
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
