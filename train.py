import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm

class InventoryDataset(Dataset):
    """Custom dataset for inventory items with augmentation"""
    
    def __init__(self, data_dir: str, transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.items = self._load_items()
        
        # Define augmentation pipeline
        self.augmentation = A.Compose([
            # Geometric transforms
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.7
            ),
            
            # Perspective and distortion
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.OpticalDistortion(distort_limit=0.2, p=0.3),
            
            # Color transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, p=0.5),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
            
            # Occlusion
            A.CoarseDropout(
                max_holes=3,
                max_height=50,
                max_width=50,
                p=0.3
            ),
        ])
    
    def _load_items(self) -> List[Dict]:
        """Load all items and their images"""
        items = []
        
        # Assume structure: data_dir/item_id/images/
        for item_dir in self.data_dir.iterdir():
            if item_dir.is_dir():
                item_id = item_dir.name
                image_dir = item_dir / 'images'
                
                if image_dir.exists():
                    images = list(image_dir.glob('*.jpg')) + \
                             list(image_dir.glob('*.png')) + \
                             list(image_dir.glob('*.jpeg'))
                    
                    for img_path in images:
                        items.append({
                            'item_id': item_id,
                            'image_path': str(img_path),
                            'label': int(item_id.split('_')[-1])  # Assuming item_id format
                        })
        
        return items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load image
        image = cv2.imread(item['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        if self.augment and self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']
        
        # Convert to PIL for CLIP preprocessing
        image = Image.fromarray(image)
        
        # Apply CLIP preprocessing
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': item['label'],
            'item_id': item['item_id']
        }

class SiameseNetwork(nn.Module):
    """Siamese network for few-shot learning"""
    
    def __init__(self, embedding_dim=512):
        super(SiameseNetwork, self).__init__()
        
        # Use CLIP as backbone
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Freeze CLIP backbone initially
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Custom head for fine-tuning
        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        # Extract CLIP features
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
        
        # Project to embedding space
        embeddings = self.projection(features.float())
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining"""
    
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = nn.functional.pairwise_distance(anchor, positive, p=2)
        distance_negative = nn.functional.pairwise_distance(anchor, negative, p=2)
        
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()

class ModelTrainer:
    """Training pipeline for inventory recognition"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = SiameseNetwork(embedding_dim=config['embedding_dim']).to(self.device)
        
        # Loss and optimizer
        self.criterion = TripletLoss(margin=config['margin'])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
    def generate_triplets(self, batch):
        """Generate triplets from batch"""
        triplets = []
        
        labels = batch['label']
        embeddings = batch['embeddings']
        
        for i in range(len(labels)):
            anchor_label = labels[i]
            anchor_embedding = embeddings[i]
            
            # Find positive (same class)
            positive_mask = labels == anchor_label
            positive_mask[i] = False  # Exclude self
            
            if positive_mask.any():
                positive_idx = torch.where(positive_mask)[0]
                positive_idx = positive_idx[torch.randint(len(positive_idx), (1,))]
                positive_embedding = embeddings[positive_idx]
                
                # Find hard negative (different class)
                negative_mask = labels != anchor_label
                
                if negative_mask.any():
                    negative_embeddings = embeddings[negative_mask]
                    
                    # Hard negative mining
                    distances = nn.functional.pairwise_distance(
                        anchor_embedding.unsqueeze(0),
                        negative_embeddings
                    )
                    
                    # Get hardest negative (closest)
                    hard_negative_idx = torch.argmin(distances)
                    negative_embedding = negative_embeddings[hard_negative_idx]
                    
                    triplets.append((anchor_embedding, positive_embedding, negative_embedding))
        
        return triplets
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                
                # Generate triplets
                batch_data = {'label': labels, 'embeddings': embeddings}
                triplets = self.generate_triplets(batch_data)
                
                if not triplets:
                    continue
                
                # Calculate loss
                loss = 0
                for anchor, positive, negative in triplets:
                    loss += self.criterion(
                        anchor.unsqueeze(0),
                        positive.unsqueeze(0),
                        negative.unsqueeze(0)
                    )
                
                if len(triplets) > 0:
                    loss = loss / len(triplets)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            embeddings_dict = {}
            
            # Collect all embeddings
            for batch in dataloader:
                images = batch['image'].to(self.device)
                labels = batch['label']
                item_ids = batch['item_id']
                
                embeddings = self.model(images)
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if label not in embeddings_dict:
                        embeddings_dict[label] = []
                    
                    embeddings_dict[label].append(embeddings[i])
            
            # Calculate accuracy using nearest neighbor
            for label, embs in embeddings_dict.items():
                for query_emb in embs:
                    min_dist = float('inf')
                    predicted_label = -1
                    
                    for other_label, other_embs in embeddings_dict.items():
                        for other_emb in other_embs:
                            if torch.equal(query_emb, other_emb):
                                continue
                                
                            dist = nn.functional.pairwise_distance(
                                query_emb.unsqueeze(0),
                                other_emb.unsqueeze(0)
                            ).item()
                            
                            if dist < min_dist:
                                min_dist = dist
                                predicted_label = other_label
                    
                    if predicted_label == label:
                        correct += 1
                    total += 1
        
        accuracy = correct / max(total, 1)
        return accuracy
    
    def train(self, train_dataloader, val_dataloader):
        """Full training loop"""
        best_accuracy = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Validate
            val_accuracy = self.validate(val_dataloader)
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(f"best_model_epoch_{epoch + 1}.pth")
                
        return best_accuracy
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

def main():
    # Configuration
    config = {
        'data_dir': '../data/training_images',
        'embedding_dim': 256,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'margin': 0.5,
        'train_split': 0.8
    }
    
    # Load CLIP preprocessing
    _, preprocess = clip.load("ViT-B/32")
    
    # Create datasets
    dataset = InventoryDataset(
        config['data_dir'],
        transform=preprocess,
        augment=True
    )
    
    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    best_accuracy = trainer.train(train_dataloader, val_dataloader)
    
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()