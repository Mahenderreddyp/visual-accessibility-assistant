from unsloth import FastVisionModel
from datasets import load_from_disk
import torch
from transformers import TextStreamer
import random

def main():
    MODEL_PATH = "models/final/vizwiz_qwen2.5_vl"
    
    print(f"--- Loading for Inference: {MODEL_PATH} ---")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_PATH,
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model)

    # Load a few random examples from the test set
    dataset = load_from_disk("data/processed/vizwiz_ready")["test"]
    
    print("\n--- Testing 3 Random Examples ---")
    for i in range(3):
        idx = random.randint(0, len(dataset)-1)
        example = dataset[idx]
        
        # Extract Image & Question from the chat format we created
        # Structure: messages -> [user_msg, assistant_msg]
        # user_msg content -> [image, text]
        user_content = example["messages"][0]["content"]
        image = user_content[0]["image"]
        question = user_content[1]["text"]
        ground_truth = example["messages"][1]["content"][0]["text"]

        print(f"\n[Example {i+1}]")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        # Prepare input for Model
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        print("Prediction: ", end="")
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
        print("-" * 50)

if __name__ == "__main__":
    main()