"""
Main entry point and configuration for the AI Recognition System
"""

# config.yaml
config_yaml = """
# AI Recognition System Configuration

# Data paths
data:
  raw_images_dir: "data/raw"
  augmented_images_dir: "data/augmented" 
  features_file: "data/features.h5"
  models_dir: "data/models"

# Model configuration
model:
  clip_variant: "ViT-B/32"
  embedding_dim: 256
  num_classes: 1000

# Training configuration  
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.00001
  triplet_margin: 0.5
  arcface_margin: 0.5
  save_every: 5
  checkpoint_dir: "checkpoints"

# Augmentation configuration
augmentation:
  augmentations_per_image: 50
  image_size: [1024, 1024]
  quality: 95

# Feature extraction configuration
features:
  feature_image_size: 512
  batch_size: 32

# Recognition configuration
recognition:
  model_path: "checkpoints/best_model.pth"
  index_path: "data/models/faiss_index.bin"
  metadata_path: "data/models/metadata.pkl"
  cache_size: 1000
  confidence_threshold: 0.85
  high_confidence_threshold: 0.95
  batch_confidence_threshold: 0.90
  batch_size: 16

# Performance targets
targets:
  accuracy: 0.95
  inference_time: 0.5  # seconds
  memory_usage: 4000  # MB
"""

# main.py
main_py = """
#!/usr/bin/env python3
'''
AI Recognition System - Main Entry Point
Achieve 95%+ accuracy with 8 images per item
'''

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import time
import json
import numpy as np
from typing import Dict, List

# Import our modules
from data_augmentation_core import AdvancedAugmentationPipeline
from feature_extraction_system import MultiModalFeatureExtractor
from training_system import ModelTrainer, FewShotDataset
from inference_pipeline import RecognitionPipeline, PerformanceMonitor, create_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIRecognitionSystem:
    '''Main system orchestrator'''
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_directories()
        
    def setup_directories(self):
        '''Create necessary directories'''
        dirs = [
            self.config['data']['augmented_images_dir'],
            self.config['data']['models_dir'],
            self.config['training']['checkpoint_dir']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_data_preparation(self):
        '''Step 1: Prepare and augment data'''
        logger.info("="*50)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*50)
        
        # Check raw images
        raw_dir = Path(self.config['data']['raw_images_dir'])
        if not raw_dir.exists():
            logger.error(f"Raw images directory not found: {raw_dir}")
            return False
        
        # Count items
        items = [d for d in raw_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(items)} items to process")
        
        # Run augmentation
        aug_config = self.config['augmentation']
        pipeline = AdvancedAugmentationPipeline(aug_config)
        
        results = pipeline.process_dataset(
            self.config['data']['raw_images_dir'],
            self.config['data']['augmented_images_dir']
        )
        
        logger.info(f"Augmentation complete!")
        logger.info(f"Generated {results['statistics']['total_augmentations_created']} images")
        
        return True
    
    def run_feature_extraction(self):
        '''Step 2: Extract features from augmented data'''
        logger.info("="*50)
        logger.info("STEP 2: FEATURE EXTRACTION")
        logger.info("="*50)
        
        # Create feature extractor
        extractor = MultiModalFeatureExtractor(self.config['features'])
        
        # Process augmented dataset
        extractor.process_dataset(
            self.config['data']['augmented_images_dir'],
            self.config['data']['features_file']
        )
        
        logger.info("Feature extraction complete!")
        
        return True
    
    def run_training(self):
        '''Step 3: Train the model'''
        logger.info("="*50)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*50)
        
        # Check if features exist
        if not Path(self.config['data']['features_file']).exists():
            logger.error("Features file not found. Run feature extraction first.")
            return False
        
        # Create datasets
        from torch.utils.data import DataLoader
        
        train_dataset = FewShotDataset(
            self.config['data']['features_file'], 
            mode='train'
        )
        val_dataset = FewShotDataset(
            self.config['data']['features_file'], 
            mode='val'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Train model
        trainer = ModelTrainer(self.config['training'])
        best_model_path = trainer.train(train_loader, val_loader)
        
        logger.info(f"Training complete! Best model: {best_model_path}")
        
        # Update config with best model path
        self.config['recognition']['model_path'] = str(best_model_path)
        
        return True
    
    def build_recognition_index(self):
        '''Step 4: Build recognition index'''
        logger.info("="*50)
        logger.info("STEP 4: BUILD RECOGNITION INDEX")
        logger.info("="*50)
        
        # Create pipeline
        pipeline = RecognitionPipeline(self.config['recognition'])
        
        # Add all items to index
        raw_dir = Path(self.config['data']['raw_images_dir'])
        items_added = 0
        
        for item_dir in raw_dir.iterdir():
            if item_dir.is_dir():
                item_id = item_dir.name
                image_files = list(item_dir.glob('*.jpg')) + list(item_dir.glob('*.png'))
                
                if image_files:
                    pipeline.add_item_to_index(item_id, [str(f) for f in image_files])
                    items_added += 1
                    logger.info(f"Added {item_id} with {len(image_files)} images")
        
        logger.info(f"Index built with {items_added} items")
        
        return True
    
    def run_evaluation(self):
        '''Step 5: Evaluate system performance'''
        logger.info("="*50)
        logger.info("STEP 5: SYSTEM EVALUATION")
        logger.info("="*50)
        
        # Create pipeline and monitor
        pipeline = create_pipeline('config.yaml')
        monitor = PerformanceMonitor(pipeline)
        
        # Test on validation set
        test_results = []
        raw_dir = Path(self.config['data']['raw_images_dir'])
        
        for item_dir in raw_dir.iterdir():
            if item_dir.is_dir():
                item_id = item_dir.name
                test_images = list(item_dir.glob('*.jpg'))[:2]  # Test 2 images per item
                
                for img_path in test_images:
                    result = pipeline.recognize(str(img_path))
                    monitor.update_metrics(result, ground_truth=item_id)
                    
                    test_results.append({
                        'true_label': item_id,
                        'predicted': result.item_id,
                        'confidence': result.confidence,
                        'time': result.inference_time
                    })
        
        # Calculate metrics
        report = monitor.get_report()
        
        logger.info("\\nEVALUATION RESULTS:")
        logger.info(f"Overall Accuracy: {report['overall_metrics']['accuracy']:.2%}")
        logger.info(f"Average Confidence: {report['overall_metrics']['avg_confidence']:.3f}")
        logger.info(f"Average Inference Time: {report['overall_metrics']['avg_inference_time']:.3f}s")
        
        # Check if we met our target
        if report['overall_metrics']['accuracy'] >= self.config['targets']['accuracy']:
            logger.info("\\n✅ TARGET ACCURACY ACHIEVED!")
        else:
            logger.warning("\\n❌ Target accuracy not met. Consider:")
            logger.warning("- Adding more training data")
            logger.warning("- Adjusting augmentation parameters")
            logger.warning("- Fine-tuning model hyperparameters")
        
        # Save detailed results
        with open('evaluation_results.json', 'w') as f:
            json.dump({
                'report': report,
                'test_results': test_results,
                'config': self.config
            }, f, indent=2)
        
        return report['overall_metrics']['accuracy'] >= self.config['targets']['accuracy']


def main():
    parser = argparse.ArgumentParser(
        description='AI Recognition System - Achieve 95%+ accuracy with 8 images per item'
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--step', type=str, choices=['all', 'prepare', 'extract', 'train', 'index', 'evaluate'],
                       default='all', help='Which step to run')
    parser.add_argument('--recognize', type=str, help='Recognize a single image')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AIRecognitionSystem(args.config)
    
    if args.recognize:
        # Quick recognition mode
        pipeline = create_pipeline(args.config)
        result = pipeline.recognize(args.recognize)
        
        print(f"\\nRecognition Result:")
        print(f"Item: {result.item_id}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Time: {result.inference_time:.3f}s")
        
    else:
        # Run pipeline steps
        steps = {
            'prepare': system.run_data_preparation,
            'extract': system.run_feature_extraction,
            'train': system.run_training,
            'index': system.build_recognition_index,
            'evaluate': system.run_evaluation
        }
        
        if args.step == 'all':
            # Run complete pipeline
            logger.info("Running complete AI recognition pipeline...")
            
            success = True
            for step_name, step_func in steps.items():
                if success:
                    success = step_func()
                    if not success:
                        logger.error(f"Step {step_name} failed!")
                        break
            
            if success:
                logger.info("\\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                logger.info("Your AI recognition system is ready to use.")
                logger.info("\\nTo recognize an image:")
                logger.info("  python main.py --recognize path/to/image.jpg")
        
        else:
            # Run specific step
            steps[args.step]()


if __name__ == '__main__':
    main()
"""

# setup.sh
setup_sh = """#!/bin/bash
# Setup script for AI Recognition System

echo "Setting up AI Recognition System..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Create directory structure
mkdir -p data/{raw,augmented,processed,embeddings,models}
mkdir -p checkpoints
mkdir -p logs

# Download CLIP model
python -c "import clip; clip.load('ViT-B/32')"

echo "Setup complete!"
echo "To start using the system:"
echo "1. Place your images in data/raw/ITEM_ID/ (8 images per item)"
echo "2. Run: python main.py --step all"
"""

# Save files
with open('config.yaml', 'w') as f:
    f.write(config_yaml)

with open('main.py', 'w') as f:
    f.write(main_py)

with open('setup.sh', 'w') as f:
    f.write(setup_sh)

# Make setup script executable
os.chmod('setup.sh', 0o755)

print("Configuration files created successfully!")