"""
Evaluate Visual Accessibility Assistant Model
Calculates accuracy on a subset of the test data.
"""

# ========================================
# Windows Compatibility Fixes
# ========================================
import os
os.environ["DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from unsloth import FastVisionModel
from datasets import load_from_disk
from tqdm import tqdm
import random
import json
from transformers import TextStreamer
from PIL import Image
import io

def is_correct(prediction, ground_truth):
    """
    Checks if prediction is 'correct enough'.
    """
    p = prediction.strip().lower()
    g = ground_truth.strip().lower()
    
    # 1. Exact match
    if p == g:
        return True
        
    # 2. Containment
    if g in p:
        return True
        
    # 3. Reverse Containment (rare, but possible for short answers)
    if p in g and len(p) > 3:
        return True
        
    return False

def main():
    # --- Configuration ---
    MODEL_PATH = "models/final/vizwiz_qwen2.5_vl_complete"
    DATASET_PATH = "data/processed/vizwiz_ready"
    NUM_SAMPLES = -1  # Set to -1 to run ALL test samples
    OUTPUT_FILE = "evaluation_results.json"
    
    print("="*60)
    print("VISUAL ACCESSIBILITY ASSISTANT - EVALUATION")
    print("="*60)
    
    # 1. Load Model
    print(f"\n[1/4] Loading model from {MODEL_PATH}...")
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            MODEL_PATH,
            load_in_4bit=True,
            use_gradient_checkpointing=True,
        )
        FastVisionModel.for_inference(model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Load Dataset
    print(f"\n[2/4] Loading test dataset...")
    try:
        dataset = load_from_disk(DATASET_PATH)["test"]
        print(f"✅ Loaded {len(dataset)} test examples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # 3. Select Samples
    if NUM_SAMPLES > 0 and NUM_SAMPLES < len(dataset):
        indices = random.sample(range(len(dataset)), NUM_SAMPLES)
        subset = dataset.select(indices)
        print(f"   Running evaluation on {NUM_SAMPLES} random samples.")
    else:
        subset = dataset
        print(f"   Running evaluation on ALL {len(dataset)} samples.")

    # 4. Run Inference
    print(f"\n[3/4] Running inference...")
    results = []
    correct_count = 0
    
    # Custom streamer to capture output silently
    class CaptureStreamer(TextStreamer):
        def __init__(self, tokenizer):
            super().__init__(tokenizer, skip_prompt=True)
            self.captured_text = ""
        def on_finalized_text(self, text, stream_end=False):
            self.captured_text += text

    for example in tqdm(subset):
        # Extract inputs
        user_content = example["messages"][0]["content"]
        image_data = user_content[0]["image"] 
        question = user_content[1]["text"]
        ground_truth = example["messages"][1]["content"][0]["text"]
        
        # --- Handle Dictionary Images ---
        if isinstance(image_data, dict):
            if "bytes" in image_data and image_data["bytes"]:
                image = Image.open(io.BytesIO(image_data["bytes"]))
            elif "path" in image_data and image_data["path"]:
                image = Image.open(image_data["path"])
            else:
                image = image_data
        else:
            image = image_data
        # -------------------------------------
        
        # Prepare for model
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        if hasattr(image, "convert"):
            image = image.convert("RGB")
            
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # --- CRITICAL FIX: Disable Truncation ---
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            # We explicitly disable truncation so huge images don't get chopped off
            truncation=False, 
        ).to("cuda")

        # Generate
        streamer = CaptureStreamer(tokenizer)
        _ = model.generate(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=64, 
            use_cache=True,
            temperature=0.1, 
            do_sample=False
        )
        
        prediction = streamer.captured_text.strip()
        
        # Check correctness
        correct = is_correct(prediction, ground_truth)
        if correct:
            correct_count += 1
            
        # Log result
        results.append({
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": correct
        })

    # 5. Report
    accuracy = (correct_count / len(subset)) * 100
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"✅ Accuracy: {accuracy:.2f}% ({correct_count}/{len(subset)})")
    
    # Show failed examples
    failures = [r for r in results if not r["correct"]]
    if failures:
        print("\n❌ Failed Examples (First 3):")
        for f in failures[:3]:
            print(f"   Q: {f['question']}")
            print(f"      Pred: {f['prediction']}")
            print(f"      True: {f['ground_truth']}")
            print("-" * 30)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Full log saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()