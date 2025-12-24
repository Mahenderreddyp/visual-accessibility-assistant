"""
Evaluate VizWiz Assistant against ground truth annotations
Calculates accuracy, precision, recall, and VizWiz-specific metrics
"""

import json
import base64
import requests
from pathlib import Path
from typing import List, Dict
import re
from collections import defaultdict
import time

# For better string matching
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("⚠️  fuzzywuzzy not installed. Install for better matching: pip install fuzzywuzzy")

class VizWizEvaluator:
    """Evaluate model against VizWiz ground truth"""
    
    def __init__(self, annotations_path: str, images_dir: str, model_name: str = "vizwiz-vl"):
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        self.model_name = model_name
        
        # Load annotations
        with open(self.annotations_path, encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        print(f"✅ Loaded {len(self.annotations)} annotations from {self.annotations_path.name}")
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', ' ', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    def check_exact_match(self, model_answer: str, ground_truth_answers: List[str]) -> bool:
        """Check if model answer exactly matches any ground truth"""
        model_norm = self.normalize_answer(model_answer)
        
        for gt in ground_truth_answers:
            gt_norm = self.normalize_answer(gt)
            if model_norm == gt_norm:
                return True
        
        return False
    
    def check_fuzzy_match(self, model_answer: str, ground_truth_answers: List[str], threshold: int = 80) -> bool:
        """Check if model answer fuzzy matches any ground truth"""
        if not FUZZYWUZZY_AVAILABLE:
            return False
        
        model_norm = self.normalize_answer(model_answer)
        
        for gt in ground_truth_answers:
            gt_norm = self.normalize_answer(gt)
            # Use token set ratio (handles word order differences)
            score = fuzz.token_set_ratio(model_norm, gt_norm)
            if score >= threshold:
                return True
        
        return False
    
    def check_contains_match(self, model_answer: str, ground_truth_answers: List[str]) -> bool:
        """Check if model answer contains any ground truth answer"""
        model_norm = self.normalize_answer(model_answer)
        
        for gt in ground_truth_answers:
            gt_norm = self.normalize_answer(gt)
            if gt_norm in model_norm or model_norm in gt_norm:
                return True
        
        return False
    
    def check_unanswerable(self, model_answer: str) -> bool:
        """Check if model correctly identified unanswerable question"""
        unanswerable_phrases = [
            'cannot answer',
            'unanswerable',
            'cannot tell',
            'unclear',
            'blurry',
            'too dark',
            'not visible',
            'cannot see',
        ]
        
        model_lower = model_answer.lower()
        return any(phrase in model_lower for phrase in unanswerable_phrases)
    
    def get_model_answer(self, image_path: Path, question: str) -> str:
        """Get model's answer for an image and question"""
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
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
    
    def evaluate(self, num_samples: int = 100, start_idx: int = 0):
        """Run evaluation on dataset"""
        
        print("\n" + "="*70)
        print(f"🔍 EVALUATING {self.model_name.upper()}")
        print("="*70)
        print(f"Samples: {num_samples} (starting from index {start_idx})")
        print(f"Images: {self.images_dir}")
        print("="*70 + "\n")
        
        # Select samples
        samples = self.annotations[start_idx:start_idx + num_samples]
        
        # Metrics
        metrics = {
            'total': 0,
            'skipped': 0,  # Image not found
            'exact_match': 0,
            'fuzzy_match': 0,
            'contains_match': 0,
            'unanswerable_correct': 0,
            'unanswerable_incorrect': 0,
            'answerable_correct': 0,
            'answerable_incorrect': 0,
        }
        
        # Detailed results
        results = []
        
        # Category breakdown
        by_answer_type = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        # Process each sample
        for i, ann in enumerate(samples, 1):
            image_path = self.images_dir / ann['image']
            
            # Skip if image not found
            if not image_path.exists():
                metrics['skipped'] += 1
                continue
            
            metrics['total'] += 1
            
            # Get ground truth
            question = ann['question']
            gt_answers = [a['answer'] for a in ann['answers']]
            answerable = ann.get('answerable', 1)
            answer_type = ann.get('answer_type', 'other')
            
            # Get model answer
            print(f"[{i}/{len(samples)}] {ann['image'][:40]:40} ", end='', flush=True)
            
            start_time = time.time()
            model_answer = self.get_model_answer(image_path, question)
            inference_time = time.time() - start_time
            
            # Check if model refused
            model_refused = self.check_unanswerable(model_answer)
            
            # Evaluate
            is_correct = False
            match_type = None
            
            if answerable == 0:
                # Ground truth is unanswerable
                if model_refused:
                    is_correct = True
                    match_type = 'unanswerable_correct'
                    metrics['unanswerable_correct'] += 1
                else:
                    metrics['unanswerable_incorrect'] += 1
            else:
                # Ground truth is answerable
                if model_refused:
                    # Model refused when it should have answered
                    metrics['answerable_incorrect'] += 1
                else:
                    # Check answer quality
                    if self.check_exact_match(model_answer, gt_answers):
                        is_correct = True
                        match_type = 'exact_match'
                        metrics['exact_match'] += 1
                        metrics['answerable_correct'] += 1
                    elif self.check_fuzzy_match(model_answer, gt_answers):
                        is_correct = True
                        match_type = 'fuzzy_match'
                        metrics['fuzzy_match'] += 1
                        metrics['answerable_correct'] += 1
                    elif self.check_contains_match(model_answer, gt_answers):
                        is_correct = True
                        match_type = 'contains_match'
                        metrics['contains_match'] += 1
                        metrics['answerable_correct'] += 1
                    else:
                        metrics['answerable_incorrect'] += 1
            
            # Update category breakdown
            by_answer_type[answer_type]['total'] += 1
            if is_correct:
                by_answer_type[answer_type]['correct'] += 1
            
            # Print result
            status = "✅" if is_correct else "❌"
            print(f"{status} ({inference_time:.1f}s)")
            
            # Store detailed result
            results.append({
                'image': ann['image'],
                'question': question,
                'ground_truth': gt_answers,
                'model_answer': model_answer,
                'answerable': answerable,
                'answer_type': answer_type,
                'correct': is_correct,
                'match_type': match_type,
                'inference_time': inference_time,
            })
        
        # Calculate final metrics
        self.print_results(metrics, by_answer_type, results)
        
        # Save results
        self.save_results(results, metrics, by_answer_type)
        
        return metrics, results
    
    def print_results(self, metrics: dict, by_answer_type: dict, results: list):
        """Print evaluation results"""
        
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)
        
        # Overall accuracy
        total_evaluated = metrics['total']
        total_correct = metrics['exact_match'] + metrics['fuzzy_match'] + metrics['contains_match'] + metrics['unanswerable_correct']
        
        overall_accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
        
        print(f"\n🎯 OVERALL PERFORMANCE")
        print(f"   Total evaluated: {total_evaluated}")
        print(f"   Skipped (not found): {metrics['skipped']}")
        print(f"   Correct: {total_correct}")
        print(f"   Accuracy: {overall_accuracy:.2f}%")
        
        # Match type breakdown
        print(f"\n📈 MATCH BREAKDOWN")
        print(f"   Exact match: {metrics['exact_match']} ({metrics['exact_match']/total_evaluated*100:.1f}%)")
        print(f"   Fuzzy match: {metrics['fuzzy_match']} ({metrics['fuzzy_match']/total_evaluated*100:.1f}%)")
        print(f"   Contains match: {metrics['contains_match']} ({metrics['contains_match']/total_evaluated*100:.1f}%)")
        
        # Answerable vs Unanswerable
        print(f"\n🔍 ANSWERABLE vs UNANSWERABLE")
        
        answerable_total = metrics['answerable_correct'] + metrics['answerable_incorrect']
        unanswerable_total = metrics['unanswerable_correct'] + metrics['unanswerable_incorrect']
        
        if answerable_total > 0:
            answerable_acc = metrics['answerable_correct'] / answerable_total * 100
            print(f"   Answerable questions: {answerable_total}")
            print(f"   Correct: {metrics['answerable_correct']} ({answerable_acc:.1f}%)")
            print(f"   Incorrect: {metrics['answerable_incorrect']}")
        
        if unanswerable_total > 0:
            unanswerable_acc = metrics['unanswerable_correct'] / unanswerable_total * 100
            print(f"   Unanswerable questions: {unanswerable_total}")
            print(f"   Correctly refused: {metrics['unanswerable_correct']} ({unanswerable_acc:.1f}%)")
            print(f"   Incorrectly answered: {metrics['unanswerable_incorrect']}")
        
        # By answer type
        if by_answer_type:
            print(f"\n📋 PERFORMANCE BY ANSWER TYPE")
            for ans_type, stats in sorted(by_answer_type.items()):
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total'] * 100
                    print(f"   {ans_type:15} {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        # Performance assessment
        print(f"\n" + "="*70)
        print("🏆 ASSESSMENT")
        print("="*70)
        
        if overall_accuracy >= 70:
            grade = "EXCELLENT ⭐⭐⭐"
            msg = "Production-ready quality!"
        elif overall_accuracy >= 50:
            grade = "GOOD ⭐⭐"
            msg = "Strong performance, minor improvements possible"
        elif overall_accuracy >= 30:
            grade = "FAIR ⭐"
            msg = "Model shows learning, needs more training"
        else:
            grade = "NEEDS IMPROVEMENT"
            msg = "Requires significant retraining"
        
        print(f"\nOverall Grade: {grade}")
        print(f"Status: {msg}")
        
        # Training recommendations
        print(f"\n💡 RECOMMENDATIONS")
        
        if overall_accuracy < 60:
            print(f"   🔄 Train longer (current: 60 steps → target: 2500 steps)")
            print(f"   📈 Expected improvement: {overall_accuracy:.0f}% → 65-75%")
        
        if metrics['answerable_incorrect'] > metrics['answerable_correct']:
            print(f"   ⚠️  Model refuses too often")
            print(f"   💡 More training will improve confidence")
        
        # Show some examples
        print(f"\n" + "="*70)
        print("✅ CORRECT EXAMPLES")
        print("="*70)
        
        correct_examples = [r for r in results if r['correct']][:5]
        for ex in correct_examples:
            print(f"\n📸 {ex['image']}")
            print(f"   Q: {ex['question']}")
            print(f"   GT: {', '.join(ex['ground_truth'][:3])}")
            print(f"   Model: {ex['model_answer'][:100]}")
            print(f"   Match: {ex['match_type']}")
        
        print(f"\n" + "="*70)
        print("❌ INCORRECT EXAMPLES")
        print("="*70)
        
        incorrect_examples = [r for r in results if not r['correct']][:5]
        for ex in incorrect_examples:
            print(f"\n📸 {ex['image']}")
            print(f"   Q: {ex['question']}")
            print(f"   GT: {', '.join(ex['ground_truth'][:3])}")
            print(f"   Model: {ex['model_answer'][:100]}")
            if ex['answerable'] == 0:
                print(f"   Note: This was unanswerable - model should have refused")
        
        print("\n" + "="*70)
    
    def save_results(self, results: list, metrics: dict, by_answer_type: dict):
        """Save detailed results to JSON"""
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output = {
            'timestamp': timestamp,
            'model': self.model_name,
            'metrics': metrics,
            'by_answer_type': dict(by_answer_type),
            'results': results,
        }
        
        filename = f"evaluation_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")


def main():
    """Run evaluation"""
    
    # Configuration
    ANNOTATIONS_PATH = r"C:\Users\mmsvl\Personal_Projects\visual-accessibility-assistant\data\raw\annotations\train.json"
    IMAGES_DIR = r"C:\Users\mmsvl\Personal_Projects\visual-accessibility-assistant\data\raw\images\train"
    MODEL_NAME = "vizwiz-vl"
    
    # Number of samples to test (start small, then increase)
    NUM_SAMPLES = 100  # Change to 500 or 1000 for comprehensive evaluation
    START_IDX = 0
    
    # Create evaluator
    evaluator = VizWizEvaluator(
        annotations_path=ANNOTATIONS_PATH,
        images_dir=IMAGES_DIR,
        model_name=MODEL_NAME
    )
    
    # Run evaluation
    metrics, results = evaluator.evaluate(
        num_samples=NUM_SAMPLES,
        start_idx=START_IDX
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()