# scripts/03_train_model_optimized.py
"""
OPTIMIZED Training for RTX 3080 - Visual Accessibility Assistant
Windows-compatible version with all optimizations except multiprocessing
"""

# ========================================
# CRITICAL: Disable torch.compile (Windows fix)
# ========================================
import os
os.environ["DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for Ampere GPUs (RTX 3080)
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

import torch
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from pathlib import Path
import platform

# Enable TF32 precision (huge speedup on RTX 3080)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """Train Qwen2.5-VL on VizWiz dataset - OPTIMIZED for RTX 3080"""
    
    # ========================================
    # Configuration - OPTIMIZED FOR RTX 3080
    # ========================================
    MODEL_ID = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
    DATASET_PATH = "data/processed/vizwiz_ready"
    OUTPUT_DIR = "models/checkpoints"
    FINAL_DIR = "models/final/vizwiz_qwen2.5_vl_complete"
    
    # Training hyperparameters - OPTIMIZED
    MAX_STEPS = 2500
    BATCH_SIZE = 4              # ⬆️ Increased from 2 (better GPU util)
    GRAD_ACCUM = 2              # ⬇️ Reduced from 4 (keep effective batch=8)
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 100          # ⬆️ Increased from 5
    MAX_GRAD_NORM = 1.0         # 🆕 Gradient clipping
    
    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 16
    
    # Logging/saving - OPTIMIZED
    LOGGING_STEPS = 10          # ⬇️ From 1
    SAVE_STEPS = 250            # ⬇️ From 20
    EVAL_STEPS = 500            # ⬇️ From 20
    
    # Windows fix: No multiprocessing for DataLoader
    IS_WINDOWS = platform.system() == "Windows"
    NUM_WORKERS = 0 if IS_WINDOWS else 2  # 🔧 Windows fix
    
    print("=" * 60)
    print("🚀 VISUAL ACCESSIBILITY ASSISTANT - OPTIMIZED TRAINING")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"GPU: RTX 3080 (10GB)")
    print(f"Platform: {platform.system()}")
    print(f"Max Steps: {MAX_STEPS}")
    print(f"Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"Optimizations: TF32, Larger Batches, Reduced Logging")
    if IS_WINDOWS:
        print(f"Note: DataLoader workers disabled (Windows multiprocessing issue)")
    print("=" * 60)
    
    # ========================================
    # Load Model with optimizations
    # ========================================
    print("\n[1/5] Loading model with optimizations...")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    print("✅ Model loaded")
    print(f"   - Quantization: 4-bit")
    print(f"   - Gradient checkpointing: Unsloth (optimized)")
    print(f"   - TF32 enabled: True (Ampere speedup)")
    
    # ========================================
    # Attach LoRA
    # ========================================
    print("\n[2/5] Attaching LoRA adapters...")
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
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
    # Setup Trainer - OPTIMIZED
    # ========================================
    print("\n[4/5] Setting up optimized trainer...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training arguments - OPTIMIZED FOR SPEED
    args = SFTConfig(
        # ===== Batch Settings (OPTIMIZED) =====
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=4,
        
        # ===== Training Steps =====
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        
        # ===== Optimization (OPTIMIZED) =====
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_grad_norm=MAX_GRAD_NORM,
        
        # ===== Precision (OPTIMIZED) =====
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        tf32=True,
        
        # ===== Logging (REDUCED OVERHEAD) =====
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",
        
        # ===== Saving (REDUCED OVERHEAD) =====
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        save_only_model=True,
        
        # ===== Evaluation (REDUCED OVERHEAD) =====
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        eval_accumulation_steps=4,
        
        # ===== DataLoader (WINDOWS COMPATIBLE) =====
        dataloader_num_workers=NUM_WORKERS,        # 🔧 0 on Windows
        dataloader_pin_memory=True,
        
        # ===== Memory Optimization =====
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # ===== Dataset =====
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        
        # ===== Stability =====
        seed=3407,
        data_seed=3407,
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
    
    print("✅ Trainer ready - OPTIMIZED FOR SPEED")
    print(f"   - Batch size: {BATCH_SIZE} → Effective: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   - Total steps: {MAX_STEPS}")
    print(f"   - Warmup: {WARMUP_STEPS} steps")
    print(f"   - Precision: {'BF16' if is_bfloat16_supported() else 'FP16'} + TF32")
    print(f"   - Gradient clipping: {MAX_GRAD_NORM}")
    print(f"   - DataLoader workers: {NUM_WORKERS}")
    print(f"   - Logging every: {LOGGING_STEPS} steps")
    print(f"   - Saving every: {SAVE_STEPS} steps")
    print(f"   - Eval every: {EVAL_STEPS} steps")
    
    # Estimate training time
    # RTX 3080: ~1.5-2s per step with optimizations
    estimated_time_hours = (MAX_STEPS * 1.75) / 3600
    print(f"\n⏱️  Estimated training time: ~{estimated_time_hours:.1f} hours")
    
    # ========================================
    # Train
    # ========================================
    print("\n[5/5] Starting optimized training...")
    print("-" * 60)
    
    try:
        trainer_stats = trainer.train()
        
        print("-" * 60)
        print("✅ Training complete!")
        print(f"   - Time: {trainer_stats.metrics['train_runtime']:.1f}s ({trainer_stats.metrics['train_runtime']/3600:.2f}h)")
        print(f"   - Final loss: {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
        print(f"   - Samples/second: {trainer_stats.metrics.get('train_samples_per_second', 'N/A'):.2f}")
        print(f"   - Steps/second: {trainer_stats.metrics.get('train_steps_per_second', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # Save Model
    # ========================================
    print("\n💾 Saving final model...")
    
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
    print(f"✅ Training time: {trainer_stats.metrics['train_runtime']/3600:.2f} hours")
    print(f"✅ Final loss: {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"✅ Model location: {FINAL_DIR}")
    print(f"✅ Checkpoints: {OUTPUT_DIR}")
    print("\n🎯 Next steps:")
    print("   1. Evaluate: python evaluate_with_gt.py")
    print("   2. Compare: Check accuracy improvement (27% → 60-70%)")
    print("   3. Quantize: python scripts/05_quantize_export.py")
    print("=" * 60)
    
    # Print optimizations used
    print("\n📊 Optimizations Applied:")
    print(f"   ✅ Batch size: 2 → {BATCH_SIZE} (2x larger)")
    print(f"   ✅ TF32 precision: Enabled (1.5x faster)")
    print(f"   ✅ Logging: 1 → {LOGGING_STEPS} steps (10x less overhead)")
    print(f"   ✅ Saving: 20 → {SAVE_STEPS} steps (12x less I/O)")
    print(f"   ✅ Gradient clipping: {MAX_GRAD_NORM} (stability)")
    print(f"   ✅ Warmup steps: 5 → {WARMUP_STEPS} (better convergence)")
    if IS_WINDOWS:
        print(f"   ⚠️  DataLoader workers: Disabled (Windows compatibility)")
    else:
        print(f"   ✅ DataLoader workers: 2 (parallel loading)")


if __name__ == "__main__":
    
    main()