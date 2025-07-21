"""
Lightweight Refiner Training Script

Trains the lightweight refinement model on hard cases identified by the
benchmark system. This model learns to improve recognition for cases
where raw CLIP+DINOv2 features struggle.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.lightweight_refiner import (
    LightweightRefiner, 
    LightweightRefinerTrainer,
    ContrastiveLoss
)
from src.feature_extraction.feature_extractor import MultiModalFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardCasesDataset(Dataset):
    """
    Dataset for training lightweight refiner on hard cases
    """
    
    def __init__(self, hard_cases_file: str, feature_extractor: MultiModalFeatureExtractor):
        """
        Initialize dataset
        
        Args:
            hard_cases_file: Path to hard cases JSON file
            feature_extractor: Feature extractor instance
        """
        self.feature_extractor = feature_extractor
        
        # Load hard cases
        with open(hard_cases_file, 'r') as f:
            self.hard_cases = json.load(f)
        
        logger.info(f"📊 Loaded {len(self.hard_cases)} hard cases for training")
        
        # Create item to index mapping
        self.item_to_idx = {}
        unique_items = set(case['true_item_id'] for case in self.hard_cases)
        for idx, item_id in enumerate(sorted(unique_items)):
            self.item_to_idx[item_id] = idx
        
        logger.info(f"🏷️  Found {len(self.item_to_idx)} unique items")
    
    def __len__(self):
        return len(self.hard_cases)
    
    def __getitem__(self, idx):
        case = self.hard_cases[idx]
        
        # Extract features
        features = self.feature_extractor.extract_all_features(case['image_path'])
        
        # Combine CLIP + DINOv2 features
        if features and 'dinov2' in features and features['dinov2'] is not None:
            combined_features = np.concatenate([features['clip'], features['dinov2']])
        else:
            # Fallback to CLIP only, padded to 1536D
            combined_features = features['clip'] if features else np.zeros(768)
            if len(combined_features) < 1536:
                padding = np.zeros(1536 - len(combined_features))
                combined_features = np.concatenate([combined_features, padding])
        
        # Get label
        label = self.item_to_idx[case['true_item_id']]
        
        return torch.FloatTensor(combined_features), torch.LongTensor([label])


def create_training_data(hard_cases_file: str, config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        hard_cases_file: Path to hard cases file
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Initialize feature extractor
    feature_config = {
        'clip_variant': config.get('clip_variant', 'ViT-L/14'),
        'dinov2_variant': config.get('dinov2_variant', 'dinov2_vitb14'),
        'feature_image_size': config.get('feature_image_size', 768),
        'batch_size': config.get('batch_size', 16)
    }
    
    feature_extractor = MultiModalFeatureExtractor(feature_config)
    
    # Create dataset
    dataset = HardCasesDataset(hard_cases_file, feature_extractor)
    
    # Split into train/val
    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders (num_workers=0 to avoid MPS tensor sharing issues)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"📚 Training set: {len(train_dataset)} samples")
    logger.info(f"🔬 Validation set: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def train_lightweight_refiner(config_path: str, hard_cases_file: str, output_path: str):
    """
    Train the lightweight refiner model
    
    Args:
        config_path: Path to configuration file
        hard_cases_file: Path to hard cases JSON file
        output_path: Output path for trained model
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    
    # Create model
    model = LightweightRefiner(
        input_dim=1536,
        hidden_dim=512,
        output_dim=256,
        dropout_rate=0.1
    )
    
    # Initialize trainer
    trainer = LightweightRefinerTrainer(model)
    
    # Create data loaders
    train_loader, val_loader = create_training_data(hard_cases_file, config)
    
    # Training parameters
    num_epochs = training_config.get('lightweight_epochs', 20)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    logger.info(f"🚀 Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            trainer.save_model(output_path)
            logger.info(f"✅ New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"🏁 Training completed! Best model saved to {output_path}")
    
    # Create FAISS index with trained model
    create_learned_index(output_path, config)


def create_learned_index(model_path: str, config: Dict):
    """
    Create FAISS index using the trained lightweight refiner
    
    Args:
        model_path: Path to trained model
        config: Configuration dictionary
    """
    logger.info("🏗️  Creating learned FAISS index...")
    
    # Load trained model
    model = LightweightRefinerTrainer.load_model(model_path)
    
    # Load raw features and create refined embeddings
    features_file = "data/features.h5"
    if not Path(features_file).exists():
        logger.error(f"❌ Features file not found: {features_file}")
        return
    
    import h5py
    import faiss
    import pickle
    
    embeddings = []
    item_ids = []
    
    with h5py.File(features_file, 'r') as hf:
        for key in hf.keys():
            if key.startswith('image_'):
                # Extract features
                clip_features = hf[key]['clip'][:]
                if 'dinov2' in hf[key]:
                    dinov2_features = hf[key]['dinov2'][:]
                    combined = np.concatenate([clip_features, dinov2_features])
                else:
                    # Pad CLIP to 1536D
                    combined = clip_features
                    if len(combined) < 1536:
                        padding = np.zeros(1536 - len(combined))
                        combined = np.concatenate([combined, padding])
                
                # Generate refined embedding
                refined = model.extract_features(combined)
                embeddings.append(refined[0])  # Remove batch dimension
                
                # Extract item_id (assuming format: image_itemid_xxx)
                parts = key.split('_')
                if len(parts) >= 2:
                    item_ids.append(parts[1])
                else:
                    item_ids.append('unknown')
    
    if not embeddings:
        logger.error("❌ No embeddings generated")
        return
    
    # Convert to numpy array
    embeddings_array = np.vstack(embeddings).astype(np.float32)
    logger.info(f"📊 Generated {embeddings_array.shape[0]} refined embeddings ({embeddings_array.shape[1]}D)")
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine similarity for normalized embeddings
    index.add(embeddings_array)
    
    # Save index
    learned_index_path = config['recognition']['learned_index_path']
    learned_metadata_path = config['recognition']['learned_metadata_path']
    
    # Create directories if needed
    Path(learned_index_path).parent.mkdir(exist_ok=True, parents=True)
    Path(learned_metadata_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Save index
    faiss.write_index(index, learned_index_path)
    
    # Save metadata
    metadata = {
        'item_ids': item_ids,
        'embedding_dim': dimension,
        'total_vectors': len(embeddings)
    }
    
    with open(learned_metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"✅ Learned index saved:")
    logger.info(f"  Index: {learned_index_path}")
    logger.info(f"  Metadata: {learned_metadata_path}")
    logger.info(f"  Vectors: {len(embeddings)}")
    logger.info(f"  Dimensions: {dimension}")


def main():
    """Main training script"""
    # Paths
    config_path = "config.yaml"
    hard_cases_file = "benchmark_results/hard_cases_for_training.json"
    output_model_path = "checkpoints/lightweight_refiner.pth"
    
    # Check if hard cases file exists
    if not Path(hard_cases_file).exists():
        logger.error(f"❌ Hard cases file not found: {hard_cases_file}")
        logger.info("💡 Run benchmark_hybrid_system.py first to identify hard cases")
        return
    
    # Create checkpoints directory
    Path(output_model_path).parent.mkdir(exist_ok=True)
    
    # Train model
    train_lightweight_refiner(config_path, hard_cases_file, output_model_path)
    
    print(f"✅ Training completed successfully!")
    print(f"🎯 Model saved to: {output_model_path}")
    print(f"🏗️  Learned FAISS index created")
    print(f"🔄 Update config.yaml to enable hybrid_mode: true")


if __name__ == "__main__":
    main()