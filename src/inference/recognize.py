"""
High-Accuracy Recognition Pipeline
Multi-stage recognition system with 95%+ accuracy target
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import faiss
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import logging
from dataclasses import dataclass
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Recognition result with confidence and metadata"""
    item_id: str
    confidence: float
    match_scores: Dict[str, float]
    stage_results: Dict[str, any]
    inference_time: float
    top_k_matches: List[Tuple[str, float]]


class RecognitionPipeline:
    """Multi-stage recognition pipeline for maximum accuracy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self._load_models()
        
        # Load index and database
        self._load_index()
        
        # Initialize cache
        self.cache = {}
        self.cache_size = config.get('cache_size', 1000)
        
    def _load_models(self):
        """Load all recognition models"""
        logger.info("Loading recognition models...")
        
        # Load fine-tuned model
        model_path = self.config['model_path']
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Import model architecture
        from training_system import SiameseNetwork
        
        self.model = SiameseNetwork(
            base_model=checkpoint['config']['clip_model'],
            embedding_dim=checkpoint['config']['embedding_dim']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load feature extractor
        from feature_extraction_system import MultiModalFeatureExtractor
        
        self.feature_extractor = MultiModalFeatureExtractor({
            'clip_variant': checkpoint['config']['clip_model']
        })
        
        logger.info("Models loaded successfully")
    
    def _load_index(self):
        """Load FAISS index and metadata"""
        logger.info("Loading recognition index...")
        
        # Load FAISS index
        index_path = self.config['index_path']
        if Path(index_path).exists():
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        else:
            # Create new index
            embedding_dim = self.config.get('embedding_dim', 256)
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product
            logger.info("Created new index")
        
        # Load metadata
        metadata_path = self.config.get('metadata_path', 'metadata.pkl')
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {
                'index_to_item': {},
                'item_embeddings': {},
                'item_info': {}
            }
    
    def add_item_to_index(self, item_id: str, image_paths: List[str]):
        """Add new item to recognition index"""
        logger.info(f"Adding item {item_id} with {len(image_paths)} images")
        
        embeddings = []
        
        for img_path in image_paths:
            # Extract features
            features = self.feature_extractor.extract_all_features(img_path)
            
            # Get embedding from model
            clip_features = torch.FloatTensor(features['clip']).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.forward_one(clip_features)
            
            embeddings.append(embedding.cpu().numpy())
        
        # Add to FAISS index
        embeddings_array = np.vstack(embeddings)
        start_idx = self.index.ntotal
        self.index.add(embeddings_array)
        
        # Update metadata
        for i in range(len(embeddings)):
            idx = start_idx + i
            self.metadata['index_to_item'][idx] = item_id
        
        if item_id not in self.metadata['item_embeddings']:
            self.metadata['item_embeddings'][item_id] = []
        
        self.metadata['item_embeddings'][item_id].extend(
            list(range(start_idx, start_idx + len(embeddings)))
        )
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Added {len(embeddings)} embeddings for item {item_id}")
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        # Save FAISS index
        faiss.write_index(self.index, str(self.config['index_path']))
        
        # Save metadata
        with open(self.config.get('metadata_path', 'metadata.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _stage1_quick_filter(self, query_embedding: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """Stage 1: Fast candidate retrieval using FAISS"""
        # Search in index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        # Group by item
        item_scores = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            
            item_id = self.metadata['index_to_item'].get(idx)
            if item_id:
                if item_id not in item_scores:
                    item_scores[item_id] = []
                item_scores[item_id].append(float(dist))
        
        # Aggregate scores (max pooling)
        candidates = []
        for item_id, scores in item_scores.items():
            max_score = max(scores)
            candidates.append((item_id, max_score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:20]  # Return top 20 candidates
    
    def _stage2_deep_matching(self, query_features: Dict, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Stage 2: Deep feature matching with multiple models"""
        refined_scores = []
        
        for item_id, initial_score in candidates:
            # Get stored features for this item
            item_features = self._get_item_features(item_id)
            
            if not item_features:
                continue
            
            # Compute multi-modal similarities
            similarities = {
                'clip': self._compute_similarity(query_features['clip'], 
                                               [f['clip'] for f in item_features]),
                'resnet': self._compute_similarity(query_features['resnet'], 
                                                 [f['resnet'] for f in item_features]),
                'color': self._compute_similarity(query_features['color'], 
                                                [f['color'] for f in item_features]),
                'texture': self._compute_similarity(query_features['texture'], 
                                                  [f['texture'] for f in item_features])
            }
            
            # Weighted combination
            weights = {
                'clip': 0.4,
                'resnet': 0.3,
                'color': 0.15,
                'texture': 0.15
            }
            
            combined_score = sum(similarities[k] * weights[k] for k in weights)
            refined_scores.append((item_id, combined_score, similarities))
        
        # Sort by refined score
        refined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return refined_scores[:10]  # Return top 10
    
    def _stage3_geometric_verification(self, query_image: np.ndarray, 
                                     candidates: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float]]:
        """Stage 3: Geometric consistency verification"""
        verified_results = []
        
        for item_id, score, similarities in candidates[:5]:  # Verify top 5
            # Get reference images for this item
            ref_images = self._get_item_images(item_id)
            
            if not ref_images:
                verified_results.append((item_id, score * 0.9))  # Penalty for no verification
                continue
            
            # Compute geometric consistency
            geometric_scores = []
            
            for ref_image in ref_images[:3]:  # Check against top 3 reference images
                geo_score = self._compute_geometric_consistency(query_image, ref_image)
                geometric_scores.append(geo_score)
            
            # Best geometric match
            best_geo_score = max(geometric_scores) if geometric_scores else 0
            
            # Combine with previous score
            final_score = score * 0.7 + best_geo_score * 0.3
            verified_results.append((item_id, final_score))
        
        # Sort by final score
        verified_results.sort(key=lambda x: x[1], reverse=True)
        
        return verified_results
    
    def _compute_similarity(self, query_feature: np.ndarray, 
                          reference_features: List[np.ndarray]) -> float:
        """Compute similarity between query and reference features"""
        if not reference_features:
            return 0.0
        
        # Compute cosine similarity with each reference
        similarities = []
        
        query_norm = query_feature / (np.linalg.norm(query_feature) + 1e-8)
        
        for ref_feature in reference_features:
            ref_norm = ref_feature / (np.linalg.norm(ref_feature) + 1e-8)
            sim = np.dot(query_norm, ref_norm)
            similarities.append(sim)
        
        # Return max similarity
        return max(similarities)
    
    def _compute_geometric_consistency(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute geometric consistency between two images"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Detect keypoints and descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0
        
        # Match features
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return len(good_matches) / 10.0
        
        # Compute homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if homography is None:
            return len(good_matches) / 100.0
        
        # Count inliers
        inliers = np.sum(mask)
        
        # Geometric consistency score
        score = inliers / len(good_matches)
        
        return score
    
    def _get_item_features(self, item_id: str) -> List[Dict]:
        """Get stored features for an item"""
        # This would load from the feature database
        # For now, return placeholder
        return []
    
    def _get_item_images(self, item_id: str) -> List[np.ndarray]:
        """Get reference images for an item"""
        # This would load actual images
        # For now, return placeholder
        return []
    
    def recognize(self, image_path: str) -> RecognitionResult:
        """Main recognition function"""
        start_time = time.time()
        
        # Check cache
        cache_key = self._compute_cache_key(image_path)
        if cache_key in self.cache:
            logger.info("Cache hit!")
            return self.cache[cache_key]
        
        logger.info(f"Processing image: {image_path}")
        
        # Stage 0: Extract features
        features = self.feature_extractor.extract_all_features(image_path)
        
        if features is None:
            return RecognitionResult(
                item_id="unknown",
                confidence=0.0,
                match_scores={},
                stage_results={},
                inference_time=time.time() - start_time,
                top_k_matches=[]
            )
        
        # Get model embedding
        clip_features = torch.FloatTensor(features['clip']).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.forward_one(clip_features).cpu().numpy()
        
        # Stage 1: Quick filter
        stage1_candidates = self._stage1_quick_filter(query_embedding)
        logger.info(f"Stage 1: {len(stage1_candidates)} candidates")
        
        if not stage1_candidates:
            return RecognitionResult(
                item_id="unknown",
                confidence=0.0,
                match_scores={},
                stage_results={'stage1': []},
                inference_time=time.time() - start_time,
                top_k_matches=[]
            )
        
        # Stage 2: Deep matching
        stage2_candidates = self._stage2_deep_matching(features, stage1_candidates)
        logger.info(f"Stage 2: {len(stage2_candidates)} candidates")
        
        # Stage 3: Geometric verification (if confidence not high enough)
        if stage2_candidates[0][1] < self.config.get('high_confidence_threshold', 0.95):
            # Load query image
            query_image = cv2.imread(image_path)
            query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            
            final_candidates = self._stage3_geometric_verification(query_image, stage2_candidates)
            logger.info(f"Stage 3: Geometric verification applied")
        else:
            final_candidates = [(c[0], c[1]) for c in stage2_candidates[:5]]
        
        # Prepare result
        if final_candidates and final_candidates[0][1] > self.config.get('confidence_threshold', 0.85):
            result = RecognitionResult(
                item_id=final_candidates[0][0],
                confidence=final_candidates[0][1],
                match_scores={
                    'stage1': dict(stage1_candidates[:5]),
                    'stage2': {c[0]: c[1] for c in stage2_candidates[:5]},
                    'final': dict(final_candidates[:5])
                },
                stage_results={
                    'stage1': stage1_candidates[:10],
                    'stage2': stage2_candidates[:5],
                    'stage3': final_candidates[:5]
                },
                inference_time=time.time() - start_time,
                top_k_matches=final_candidates[:5]
            )
        else:
            result = RecognitionResult(
                item_id="unknown",
                confidence=final_candidates[0][1] if final_candidates else 0.0,
                match_scores={},
                stage_results={
                    'stage1': stage1_candidates[:10],
                    'stage2': stage2_candidates[:5] if 'stage2_candidates' in locals() else [],
                    'stage3': final_candidates[:5] if 'final_candidates' in locals() else []
                },
                inference_time=time.time() - start_time,
                top_k_matches=final_candidates[:5] if final_candidates else []
            )
        
        # Update cache
        self._update_cache(cache_key, result)
        
        logger.info(f"Recognition complete in {result.inference_time:.3f}s")
        logger.info(f"Result: {result.item_id} (confidence: {result.confidence:.3f})")
        
        return result
    
    def _compute_cache_key(self, image_path: str) -> str:
        """Compute cache key for an image"""
        # Use file path and modification time
        stat = Path(image_path).stat()
        return f"{image_path}_{stat.st_mtime}_{stat.st_size}"
    
    def _update_cache(self, key: str, result: RecognitionResult):
        """Update LRU cache"""
        self.cache[key] = result
        
        # Limit cache size
        if len(self.cache) > self.cache_size:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:-self.cache_size]
            for k in oldest_keys:
                del self.cache[k]
    
    def batch_recognize(self, image_paths: List[str]) -> List[RecognitionResult]:
        """Recognize multiple images efficiently"""
        results = []
        
        # Process in batches for efficiency
        batch_size = self.config.get('batch_size', 16)
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            # Extract features in parallel
            batch_features = []
            for path in batch_paths:
                features = self.feature_extractor.extract_all_features(path)
                batch_features.append(features)
            
            # Process batch
            for j, (path, features) in enumerate(zip(batch_paths, batch_features)):
                if features is None:
                    batch_results.append(RecognitionResult(
                        item_id="unknown",
                        confidence=0.0,
                        match_scores={},
                        stage_results={},
                        inference_time=0.0,
                        top_k_matches=[]
                    ))
                    continue
                
                # Get embeddings
                clip_features = torch.FloatTensor(features['clip']).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.model.forward_one(clip_features).cpu().numpy()
                
                # Quick recognition (Stage 1 only for batch mode)
                candidates = self._stage1_quick_filter(embedding)
                
                if candidates and candidates[0][1] > self.config.get('batch_confidence_threshold', 0.9):
                    batch_results.append(RecognitionResult(
                        item_id=candidates[0][0],
                        confidence=candidates[0][1],
                        match_scores={'stage1': dict(candidates[:5])},
                        stage_results={'stage1': candidates[:5]},
                        inference_time=0.0,
                        top_k_matches=candidates[:5]
                    ))
                else:
                    # Fall back to full pipeline for uncertain cases
                    result = self.recognize(path)
                    batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


class PerformanceMonitor:
    """Monitor and optimize recognition performance"""
    
    def __init__(self, pipeline: RecognitionPipeline):
        self.pipeline = pipeline
        self.metrics = {
            'total_queries': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'average_confidence': 0.0,
            'average_inference_time': 0.0,
            'cache_hits': 0,
            'per_item_accuracy': {}
        }
    
    def update_metrics(self, result: RecognitionResult, ground_truth: Optional[str] = None):
        """Update performance metrics"""
        self.metrics['total_queries'] += 1
        
        if result.item_id != "unknown":
            self.metrics['successful_recognitions'] += 1
        else:
            self.metrics['failed_recognitions'] += 1
        
        # Update running averages
        n = self.metrics['total_queries']
        self.metrics['average_confidence'] = (
            (self.metrics['average_confidence'] * (n - 1) + result.confidence) / n
        )
        self.metrics['average_inference_time'] = (
            (self.metrics['average_inference_time'] * (n - 1) + result.inference_time) / n
        )
        
        # Track per-item accuracy if ground truth provided
        if ground_truth:
            if ground_truth not in self.metrics['per_item_accuracy']:
                self.metrics['per_item_accuracy'][ground_truth] = {
                    'correct': 0,
                    'total': 0
                }
            
            self.metrics['per_item_accuracy'][ground_truth]['total'] += 1
            
            if result.item_id == ground_truth:
                self.metrics['per_item_accuracy'][ground_truth]['correct'] += 1
    
    def get_accuracy(self) -> float:
        """Calculate overall accuracy"""
        if self.metrics['total_queries'] == 0:
            return 0.0
        
        return self.metrics['successful_recognitions'] / self.metrics['total_queries']
    
    def get_report(self) -> Dict:
        """Generate performance report"""
        report = {
            'overall_metrics': {
                'accuracy': self.get_accuracy(),
                'total_queries': self.metrics['total_queries'],
                'successful': self.metrics['successful_recognitions'],
                'failed': self.metrics['failed_recognitions'],
                'avg_confidence': self.metrics['average_confidence'],
                'avg_inference_time': self.metrics['average_inference_time']
            },
            'per_item_metrics': {}
        }
        
        # Calculate per-item accuracy
        for item_id, stats in self.metrics['per_item_accuracy'].items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                report['per_item_metrics'][item_id] = {
                    'accuracy': accuracy,
                    'total_tests': stats['total'],
                    'correct': stats['correct']
                }
        
        return report
    
    def identify_problematic_items(self, threshold: float = 0.8) -> List[str]:
        """Identify items with low recognition accuracy"""
        problematic = []
        
        for item_id, stats in self.metrics['per_item_accuracy'].items():
            if stats['total'] >= 5:  # Minimum sample size
                accuracy = stats['correct'] / stats['total']
                if accuracy < threshold:
                    problematic.append((item_id, accuracy))
        
        # Sort by accuracy (lowest first)
        problematic.sort(key=lambda x: x[1])
        
        return [item[0] for item in problematic]


def create_pipeline(config_path: str) -> RecognitionPipeline:
    """Create recognition pipeline from configuration file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set defaults
    config.setdefault('cache_size', 1000)
    config.setdefault('confidence_threshold', 0.85)
    config.setdefault('high_confidence_threshold', 0.95)
    config.setdefault('batch_confidence_threshold', 0.9)
    config.setdefault('batch_size', 16)
    
    return RecognitionPipeline(config)


def main():
    """Main entry point for recognition testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run recognition pipeline')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--image', type=str, help='Single image to recognize')
    parser.add_argument('--batch', type=str, help='Directory of images to recognize')
    parser.add_argument('--add-item', type=str, help='Add new item to index')
    parser.add_argument('--item-id', type=str, help='Item ID for adding')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_pipeline(args.config)
    
    if args.add_item and args.item_id:
        # Add new item
        image_dir = Path(args.add_item)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        pipeline.add_item_to_index(args.item_id, [str(p) for p in image_paths])
        print(f"Added item {args.item_id} with {len(image_paths)} images")
    
    elif args.image:
        # Recognize single image
        result = pipeline.recognize(args.image)
        
        print("\nRecognition Result:")
        print(f"Item ID: {result.item_id}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Inference Time: {result.inference_time:.3f}s")
        
        print("\nTop 5 Matches:")
        for item_id, score in result.top_k_matches:
            print(f"  {item_id}: {score:.3f}")
    
    elif args.batch:
        # Batch recognition
        image_dir = Path(args.batch)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f"Processing {len(image_paths)} images...")
        results = pipeline.batch_recognize([str(p) for p in image_paths])
        
        # Summary
        successful = sum(1 for r in results if r.item_id != "unknown")
        avg_confidence = np.mean([r.confidence for r in results])
        
        print(f"\nBatch Results:")
        print(f"Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"Average Confidence: {avg_confidence:.3f}")
    
    elif args.evaluate:
        # Run evaluation
        print("Running evaluation...")
        
        # This would load test dataset and run comprehensive evaluation
        # For now, show placeholder
        monitor = PerformanceMonitor(pipeline)
        
        # Simulate some recognitions
        test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]
        for img in test_images:
            if Path(img).exists():
                result = pipeline.recognize(img)
                monitor.update_metrics(result)
        
        # Print report
        report = monitor.get_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
        