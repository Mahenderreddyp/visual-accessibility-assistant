"""Test VizWiz Assistant with images"""

import base64
import requests
import sys
from pathlib import Path

def test_image(image_path: str, question: str):
    """Test vision model with an image"""
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"📸 Image: {Path(image_path).name}")
    print(f"❓ Question: {question}")
    print(f"{'='*60}")
    print("🔍 Analyzing...\n")
    
    # Read and encode image as base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Call Ollama API
    try:
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
            result = response.json()
            answer = result.get('response', '').strip()
            
            print(f"💬 Answer:\n{answer}\n")
            print(f"{'='*60}\n")
            
            return answer
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama!")
        print("   Make sure Ollama is running:")
        print("   > ollama serve")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test with VizWiz image
    image_path = r"C:\Users\mmsvl\Personal_Projects\visual-accessibility-assistant\data\raw\images\train\VizWiz_train_00000029.jpg"
    
    # Test multiple questions
    questions = [
        "What do you see in this image?",
        "Describe this image in detail.",
        "What is this?",
    ]
    
    for question in questions:
        test_image(image_path, question)