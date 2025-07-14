#!/usr/bin/env python3
"""
GPU-Accelerated Training Script for AI Recognition System
Uses the enhanced AdvancedModelTrainer with anti-overfitting strategies
"""

import yaml
import argparse
from pathlib import Path
import logging
from src.training.modletraining import AdvancedModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['training']
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Train AI Recognition Model with GPU Acceleration')
    parser.add_argument('--config', type=str, default='config_temp.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--features', type=str, default='data/features.h5',
                       help='Path to features HDF5 file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config['checkpoint_dir'] = str(checkpoint_dir)
    
    # Check if features file exists
    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Please run feature extraction first:")
        logger.info("python main.py --step extract")
        return
    
    logger.info("🚀 Starting GPU-Accelerated Model Training")
    logger.info(f"Features file: {features_path}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Configuration: {config}")
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(config=config, features_file=str(features_path))
    
    # Start training
    try:
        best_model_path = trainer.train(str(features_path))
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Best model saved to: {best_model_path}")
        
        # Print training statistics
        stats = trainer.training_stats
        if stats.get('best_val_accuracy'):
            logger.info(f"Best validation accuracy: {stats['best_val_accuracy']:.4f}")
        
        if stats.get('cv_mean'):
            logger.info(f"Cross-validation mean: {stats['cv_mean']:.4f} ± {stats['cv_std']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()