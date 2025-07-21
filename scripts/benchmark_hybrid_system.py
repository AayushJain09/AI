"""
Hybrid Recognition System Benchmarking and Hard Case Analysis

This script evaluates the performance of the hybrid recognition system,
identifies hard cases where raw features struggle, and provides insights
for training the lightweight refinement model.
"""

import os
import sys
import json
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.recognize import RecognitionPipeline, RecognitionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark test result"""
    image_path: str
    true_item_id: str
    predicted_item_id: str
    confidence: float
    correct: bool
    inference_time: float
    used_refinement: bool
    stage_results: Dict
    hard_case: bool = False
    

class HybridSystemBenchmark:
    """
    Comprehensive benchmarking system for hybrid recognition pipeline
    """
    
    def __init__(self, config_path: str):
        """
        Initialize benchmark system
        
        Args:
            config_path: Path to system configuration file
        """
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize recognition pipeline
        recognition_config = self.config.get('recognition', {})
        self.pipeline = RecognitionPipeline(recognition_config)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.hard_cases: List[BenchmarkResult] = []
        
        logger.info(f"🚀 HybridSystemBenchmark initialized")
        logger.info(f"  Hybrid mode: {self.pipeline.hybrid_mode}")
        logger.info(f"  Lightweight model loaded: {self.pipeline.lightweight_model is not None}")
    
    def create_test_dataset(self, data_dir: str) -> List[Tuple[str, str]]:
        """
        Create test dataset from raw image directory
        
        Args:
            data_dir: Path to raw images directory
            
        Returns:
            List of (image_path, item_id) tuples
        """
        test_cases = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"❌ Data directory not found: {data_dir}")
            return test_cases
        
        # Iterate through item directories
        for item_dir in data_path.iterdir():
            if item_dir.is_dir():
                item_id = item_dir.name
                
                # Find image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                images = []
                
                for ext in image_extensions:
                    images.extend(list(item_dir.glob(f"*{ext}")))
                    images.extend(list(item_dir.glob(f"*{ext.upper()}")))
                
                # Add images to test set (use subset for faster testing)
                for image_path in images[:3]:  # Test with first 3 images per item
                    test_cases.append((str(image_path), item_id))
        
        logger.info(f"📊 Created test dataset: {len(test_cases)} test cases from {len(set(case[1] for case in test_cases))} items")
        return test_cases
    
    def run_benchmark(self, test_cases: List[Tuple[str, str]]) -> Dict:
        """
        Run comprehensive benchmark on test cases
        
        Args:
            test_cases: List of (image_path, true_item_id) tuples
            
        Returns:
            Benchmark results summary
        """
        logger.info(f"🧪 Starting benchmark with {len(test_cases)} test cases...")
        
        correct_predictions = 0
        total_inference_time = 0.0
        confidence_scores = []
        
        for i, (image_path, true_item_id) in enumerate(test_cases):
            logger.info(f"🖼️  Testing {i+1}/{len(test_cases)}: {Path(image_path).name}")
            
            # Run recognition
            start_time = time.time()
            try:
                result = self.pipeline.recognize(image_path)
                inference_time = time.time() - start_time
                
                # Determine if prediction is correct
                correct = result.item_id == true_item_id
                if correct:
                    correct_predictions += 1
                
                # Check if refinement was used (if hybrid mode)
                used_refinement = False
                if hasattr(self.pipeline, 'hybrid_stats'):
                    # Check if this query used refinement (approximate)
                    used_refinement = self.pipeline.hybrid_stats['refined'] > len(self.results)
                
                # Identify hard cases (low confidence or incorrect prediction)
                hard_case = (result.confidence < 0.85) or (not correct and result.confidence > 0.5)
                
                # Store result
                benchmark_result = BenchmarkResult(
                    image_path=image_path,
                    true_item_id=true_item_id,
                    predicted_item_id=result.item_id,
                    confidence=result.confidence,
                    correct=correct,
                    inference_time=inference_time,
                    used_refinement=used_refinement,
                    stage_results=result.stage_results,
                    hard_case=hard_case
                )
                
                self.results.append(benchmark_result)
                if hard_case:
                    self.hard_cases.append(benchmark_result)
                
                total_inference_time += inference_time
                confidence_scores.append(result.confidence)
                
                # Log result
                status = "✅" if correct else "❌"
                refinement_status = "🔄" if used_refinement else "⚡"
                logger.info(f"  {status} {refinement_status} {result.item_id} ({result.confidence:.3f}) [{inference_time:.2f}s]")
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_path}: {e}")
                continue
            
            # Progress update every 50 cases
            if (i + 1) % 50 == 0:
                current_accuracy = correct_predictions / len(self.results)
                logger.info(f"📊 Progress: {i+1}/{len(test_cases)}, Accuracy: {current_accuracy:.3f}")
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats()
        
        logger.info("🏁 Benchmark completed!")
        self._print_summary(summary)
        
        return summary
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculate comprehensive summary statistics"""
        if not self.results:
            return {}
        
        # Basic accuracy metrics
        total_cases = len(self.results)
        correct_cases = len([r for r in self.results if r.correct])
        accuracy = correct_cases / total_cases
        
        # Confidence statistics
        confidences = [r.confidence for r in self.results]
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Performance statistics
        inference_times = [r.inference_time for r in self.results]
        avg_inference_time = np.mean(inference_times)
        
        # Hybrid system statistics
        refinement_used = len([r for r in self.results if r.used_refinement])
        refinement_rate = refinement_used / total_cases
        
        # Hard case analysis
        hard_cases_count = len(self.hard_cases)
        hard_case_rate = hard_cases_count / total_cases
        
        # Accuracy by confidence ranges
        high_conf_results = [r for r in self.results if r.confidence > 0.9]
        medium_conf_results = [r for r in self.results if 0.7 <= r.confidence <= 0.9]
        low_conf_results = [r for r in self.results if r.confidence < 0.7]
        
        high_conf_accuracy = len([r for r in high_conf_results if r.correct]) / len(high_conf_results) if high_conf_results else 0
        medium_conf_accuracy = len([r for r in medium_conf_results if r.correct]) / len(medium_conf_results) if medium_conf_results else 0
        low_conf_accuracy = len([r for r in low_conf_results if r.correct]) / len(low_conf_results) if low_conf_results else 0
        
        # Get hybrid stats from pipeline
        hybrid_stats = {}
        if hasattr(self.pipeline, 'get_hybrid_stats'):
            hybrid_stats = self.pipeline.get_hybrid_stats()
        
        summary = {
            'total_test_cases': total_cases,
            'correct_predictions': correct_cases,
            'overall_accuracy': accuracy,
            'average_confidence': avg_confidence,
            'confidence_std': std_confidence,
            'average_inference_time': avg_inference_time,
            'hard_cases_count': hard_cases_count,
            'hard_case_rate': hard_case_rate,
            'refinement_usage_rate': refinement_rate,
            'confidence_breakdown': {
                'high_confidence': {
                    'count': len(high_conf_results),
                    'accuracy': high_conf_accuracy,
                    'range': '> 0.9'
                },
                'medium_confidence': {
                    'count': len(medium_conf_results),
                    'accuracy': medium_conf_accuracy,
                    'range': '0.7 - 0.9'
                },
                'low_confidence': {
                    'count': len(low_conf_results),
                    'accuracy': low_conf_accuracy,
                    'range': '< 0.7'
                }
            },
            'hybrid_stats': hybrid_stats
        }
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print formatted benchmark summary"""
        print("\n" + "="*80)
        print("🏆 HYBRID RECOGNITION SYSTEM BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\n📊 Overall Performance:")
        print(f"  Total test cases: {summary['total_test_cases']}")
        print(f"  Correct predictions: {summary['correct_predictions']}")
        print(f"  Overall accuracy: {summary['overall_accuracy']:.3f} ({summary['overall_accuracy']*100:.1f}%)")
        print(f"  Average confidence: {summary['average_confidence']:.3f} ± {summary['confidence_std']:.3f}")
        print(f"  Average inference time: {summary['average_inference_time']:.3f}s")
        
        print(f"\n🔄 Hybrid System Usage:")
        print(f"  Refinement usage rate: {summary['refinement_usage_rate']:.3f} ({summary['refinement_usage_rate']*100:.1f}%)")
        
        if summary['hybrid_stats']:
            hs = summary['hybrid_stats']
            print(f"  Raw-only decisions: {hs.get('raw_only', 0)} ({hs.get('raw_only_pct', 0):.1f}%)")
            print(f"  Refined decisions: {hs.get('refined', 0)} ({hs.get('refined_pct', 0):.1f}%)")
        
        print(f"\n📈 Performance by Confidence Range:")
        for conf_range, stats in summary['confidence_breakdown'].items():
            print(f"  {conf_range.replace('_', ' ').title()} {stats['range']}:")
            print(f"    Count: {stats['count']}")
            print(f"    Accuracy: {stats['accuracy']:.3f} ({stats['accuracy']*100:.1f}%)")
        
        print(f"\n🎯 Hard Cases Analysis:")
        print(f"  Hard cases identified: {summary['hard_cases_count']}")
        print(f"  Hard case rate: {summary['hard_case_rate']:.3f} ({summary['hard_case_rate']*100:.1f}%)")
        
        print("\n" + "="*80)
    
    def analyze_hard_cases(self) -> Dict:
        """Analyze hard cases for insights"""
        if not self.hard_cases:
            logger.info("ℹ️  No hard cases found to analyze")
            return {}
        
        logger.info(f"🔍 Analyzing {len(self.hard_cases)} hard cases...")
        
        # Group by failure type
        low_confidence_cases = [h for h in self.hard_cases if h.confidence < 0.7]
        wrong_predictions = [h for h in self.hard_cases if not h.correct]
        ambiguous_cases = [h for h in self.hard_cases if h.confidence > 0.5 and not h.correct]
        
        # Item-level analysis
        item_errors = defaultdict(list)
        for case in wrong_predictions:
            item_errors[case.true_item_id].append(case)
        
        # Most problematic items
        problematic_items = sorted(item_errors.items(), key=lambda x: len(x[1]), reverse=True)
        
        analysis = {
            'total_hard_cases': len(self.hard_cases),
            'low_confidence_cases': len(low_confidence_cases),
            'wrong_predictions': len(wrong_predictions),
            'ambiguous_cases': len(ambiguous_cases),
            'problematic_items': problematic_items[:10],  # Top 10 problematic items
            'hard_case_examples': self.hard_cases[:20]  # First 20 for training data
        }
        
        print(f"\n🔍 HARD CASES ANALYSIS:")
        print(f"  Low confidence cases: {len(low_confidence_cases)}")
        print(f"  Wrong predictions: {len(wrong_predictions)}")
        print(f"  Ambiguous cases: {len(ambiguous_cases)}")
        
        if problematic_items:
            print(f"\n🚨 Most problematic items:")
            for item_id, cases in problematic_items[:5]:
                print(f"    {item_id}: {len(cases)} errors")
        
        return analysis
    
    def save_results(self, output_path: str):
        """Save benchmark results to file"""
        results_data = {
            'benchmark_summary': self._calculate_summary_stats(),
            'hard_cases_analysis': self.analyze_hard_cases(),
            'detailed_results': [
                {
                    'image_path': r.image_path,
                    'true_item_id': r.true_item_id,
                    'predicted_item_id': r.predicted_item_id,
                    'confidence': r.confidence,
                    'correct': r.correct,
                    'inference_time': r.inference_time,
                    'used_refinement': r.used_refinement,
                    'hard_case': r.hard_case
                }
                for r in self.results
            ]
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_path}")
    
    def export_hard_cases_for_training(self, output_dir: str):
        """Export hard cases for training the lightweight refiner"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create training data file
        training_cases = []
        for case in self.hard_cases:
            training_cases.append({
                'image_path': case.image_path,
                'true_item_id': case.true_item_id,
                'confidence': case.confidence,
                'reason': 'hard_case'
            })
        
        training_file = output_path / 'hard_cases_for_training.json'
        with open(training_file, 'w') as f:
            json.dump(training_cases, f, indent=2)
        
        logger.info(f"📝 Exported {len(training_cases)} hard cases for training to {training_file}")
        return str(training_file)


def main():
    """Main benchmarking script"""
    # Configuration
    config_path = "config.yaml"
    data_dir = "data/raw"
    output_dir = "benchmark_results"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = HybridSystemBenchmark(config_path)
    
    # Create test dataset
    test_cases = benchmark.create_test_dataset(data_dir)
    
    if not test_cases:
        logger.error("❌ No test cases found. Check data directory.")
        return
    
    # Run benchmark
    timestamp = int(time.time())
    results = benchmark.run_benchmark(test_cases)
    
    # Save results
    results_file = f"{output_dir}/benchmark_results_{timestamp}.json"
    benchmark.save_results(results_file)
    
    # Export hard cases for training
    training_file = benchmark.export_hard_cases_for_training(output_dir)
    
    print(f"\n✅ Benchmark completed successfully!")
    print(f"📊 Results saved to: {results_file}")
    print(f"📝 Hard cases exported to: {training_file}")


if __name__ == "__main__":
    main()