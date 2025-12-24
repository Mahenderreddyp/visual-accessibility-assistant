# scripts/03_train_model.py
"""
Train Qwen2.5-VL on VizWiz dataset for Visual Accessibility Assistant
"""

# ========================================
# CRITICAL: Disable torch.compile (Windows fix)
# ========================================
import os
os.environ["DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from pathlib import Path


def main():
    """Train Qwen2.5-VL on VizWiz dataset"""
    
    # ========================================
    # Configuration
    # ========================================
    MODEL_ID = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
    DATASET_PATH = "data/processed/vizwiz_ready"
    OUTPUT_DIR = "models/checkpoints"
    FINAL_DIR = "models/final/vizwiz_qwen2.5_vl"
    
    # Training hyperparameters
    EPOCHS = 1
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    MAX_STEPS = 60  # For quick test, increase to 2500 for full training
    
    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 16
    
    print("=" * 60)
    print("VISUAL ACCESSIBILITY ASSISTANT - TRAINING")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max Steps: {MAX_STEPS}")
    print("=" * 60)
    
    # ========================================
    # Load Model
    # ========================================
    print("\n[1/5] Loading model...")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    print("✅ Model loaded")
    print(f"   - Quantization: 4-bit")
    print(f"   - Gradient checkpointing: Enabled")
    
    # ========================================
    # Attach LoRA
    # ========================================
    print("\n[2/5] Attaching LoRA adapters...")
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,     # Fine-tune vision encoder
        finetune_language_layers=True,   # Fine-tune language model
        finetune_attention_modules=True, # Fine-tune attention
        finetune_mlp_modules=True,       # Fine-tune MLP
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # IMPORTANT: Set to training mode
    FastVisionModel.for_training(model)
    
    # Calculate trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print("✅ LoRA attached")
    print(f"   - r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"   - Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ========================================
    # Load Dataset
    # ========================================
    print("\n[3/5] Loading dataset...")
    
    if not Path(DATASET_PATH).exists():
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("   Run: python scripts/02_process_vizwiz.py")
        return
    
    dataset = load_from_disk(DATASET_PATH)
    
    print("✅ Dataset loaded")
    print(f"   - Train: {len(dataset['train'])} examples")
    print(f"   - Test: {len(dataset['test'])} examples")
    
    # ========================================
    # Setup Trainer
    # ========================================
    print("\n[4/5] Setting up trainer...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training arguments
    args = SFTConfig(
        # Batch settings
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        
        # Training steps
        max_steps=MAX_STEPS,
        warmup_steps=5,
        
        # Optimization
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        
        # Precision
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        
        # Logging
        logging_steps=1,
        report_to="none",
        
        # Saving
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=20,
        
        # Dataset
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=args,
    )
    
    print("✅ Trainer ready")
    print(f"   - Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   - Total steps: {MAX_STEPS}")
    print(f"   - Precision: {'BF16' if is_bfloat16_supported() else 'FP16'}")
    
    # ========================================
    # Train
    # ========================================
    print("\n[5/5] Starting training...")
    print("-" * 60)
    
    try:
        trainer_stats = trainer.train()
        
        print("-" * 60)
        print("✅ Training complete!")
        print(f"   - Time: {trainer_stats.metrics['train_runtime']:.1f}s")
        print(f"   - Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # Save Model
    # ========================================
    print("\n💾 Saving model...")
    
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    
    print(f"✅ Model saved to: {FINAL_DIR}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Status: Success")
    print(f"✅ Steps completed: {MAX_STEPS}")
    print(f"✅ Model location: {FINAL_DIR}")
    print(f"✅ Checkpoints: {OUTPUT_DIR}")
    print("\n🎯 Next steps:")
    print("   1. Evaluate: python scripts/04_evaluate_model.py")
    print("   2. Quantize: python scripts/05_quantize_export.py")
    print("=" * 60)


if __name__ == "__main__":
    main()