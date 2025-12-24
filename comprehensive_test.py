"""Comprehensive testing on VizWiz images"""

import base64
import requests
from pathlib import Path
import json
from datetime import datetime

def test_image(image_path: Path, question: str):
    """Test single image"""
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'vizwiz-vl',
                'prompt': question,
                'images': [image_data],
                'stream': False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            return f"ERROR: {response.status_code}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def evaluate_quality(answer: str) -> dict:
    """Simple quality evaluation"""
    score = {
        'length': len(answer.split()),
        'has_description': any(word in answer.lower() for word in ['is', 'are', 'shows', 'contains']),
        'refuses': any(phrase in answer.lower() for phrase in ['cannot', "can't", 'unable', 'unclear', 'blurry']),
        'hallucination': any(phrase in answer.lower() for phrase in ['my vision', 'i was', 'when i']),
    }
    return score

def main():
    # Setup
    train_dir = Path(r"C:\Users\mmsvl\Personal_Projects\visual-accessibility-assistant\data\raw\images\train")
    
    # Test questions (typical VizWiz questions)
    questions = [
        "What is this?",
        "What product is this?",
        "What flavor is this?",
        "What color is this?",
        "Describe this image.",
    ]
    
    # Get 20 random images
    import random
    all_images = list(train_dir.glob("*.jpg"))
    test_images = random.sample(all_images, min(20, len(all_images)))
    
    print("="*70)
    print(f"COMPREHENSIVE TEST - VizWiz Assistant")
    print(f"Testing {len(test_images)} images with {len(questions)} questions each")
    print("="*70)
    
    results = []
    total_tests = len(test_images) * len(questions)
    current = 0
    
    for img_path in test_images:
        print(f"\n📸 Image: {img_path.name}")
        print("-" * 70)
        
        for question in questions:
            current += 1
            print(f"[{current}/{total_tests}] {question}")
            
            answer = test_image(img_path, question)
            quality = evaluate_quality(answer)
            
            # Show first 100 chars
            preview = answer[:100] + "..." if len(answer) > 100 else answer
            print(f"   → {preview}")
            
            results.append({
                'image': img_path.name,
                'question': question,
                'answer': answer,
                'quality': quality
            })
        
        print("-" * 70)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    avg_length = sum(r['quality']['length'] for r in results) / len(results)
    refusal_rate = sum(r['quality']['refuses'] for r in results) / len(results) * 100
    hallucination_rate = sum(r['quality']['hallucination'] for r in results) / len(results) * 100
    
    print(f"\n📊 Statistics:")
    print(f"   Total tests: {len(results)}")
    print(f"   Average answer length: {avg_length:.1f} words")
    print(f"   Refusal rate: {refusal_rate:.1f}% (says 'cannot answer')")
    print(f"   Hallucination rate: {hallucination_rate:.1f}% (says 'I' or 'my')")
    
    # Quality assessment
    print(f"\n✅ Quality Assessment:")
    if avg_length > 15:
        print(f"   ✅ Good detail (avg {avg_length:.0f} words)")
    else:
        print(f"   ⚠️  Short answers (avg {avg_length:.0f} words)")
    
    if refusal_rate < 30:
        print(f"   ✅ Good confidence ({refusal_rate:.0f}% refusals)")
    else:
        print(f"   ⚠️  High refusal rate ({refusal_rate:.0f}%)")
    
    if hallucination_rate < 10:
        print(f"   ✅ Low hallucination ({hallucination_rate:.0f}%)")
    else:
        print(f"   ⚠️  Hallucinating personal experiences ({hallucination_rate:.0f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()