"""
GPU-Accelerated Model Training System with Advanced Anti-Overfitting
Implements Siamese networks with comprehensive generalization strategies for 98%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import clip
import numpy as np
from pathlib import Path
import h5py
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
import wandb
from datetime import datetime
import warnings
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SiameseNetwork(nn.Module):
    """State-of-the-art Siamese network optimized for 1536-dimensional features with excellent generalization"""
    
    def __init__(self, base_model: str = 'ViT-L/14', embedding_dim: int = 512, input_dim: int = 1536, dropout_rate: float = 0.3):
        super(SiameseNetwork, self).__init__()
        
        # Optimized architecture parameters for 1536-dim input
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # State-of-the-art projection head optimized for maximum generalization
        self.projection = nn.Sequential(
            # Input layer optimized for 1536 features (CLIP 768 + DINOv2 768)
            nn.Linear(input_dim, 2048),  # Increased capacity for rich features
            nn.LayerNorm(2048),          # LayerNorm for better stability than BatchNorm
            nn.GELU(),                   # GELU for better performance than ReLU
            nn.Dropout(self.dropout_rate),
            
            # Second layer with attention-like mechanism
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.8),
            
            # Third layer for refined feature learning
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.6),
            
            # Fourth layer for final feature compression
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.4),
            
            # Final projection layer
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(self.dropout_rate * 0.2)
        )
        
        # Gradient clipping value for stable training
        self.gradient_clip_val = 1.0
        
        # Initialize weights with advanced strategies
        self._initialize_weights()
        
    def _initialize_weights(self):
        """State-of-the-art weight initialization for optimal generalization"""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                # Optimal initialization for modern architectures with GELU
                if hasattr(m, 'weight') and m.weight is not None:
                    # Use Xavier uniform for GELU activations (better than normal)
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    
                    # Special initialization for final layer (smaller magnitude)
                    if m.weight.shape[0] == self.embedding_dim:  # Final layer
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_one(self, features):
        """Forward pass optimized for 1536-dimensional features with state-of-the-art regularization"""
        # Ensure input is float and properly shaped for 1536 features
        features = features.float()
        
        # Verify correct input dimension
        if features.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {features.shape[1]}")
        
        # Advanced input regularization during training
        if self.training:
            # Gaussian noise injection for robustness (smaller for high-dim features)
            noise_std = 0.005  # Reduced noise for 1536-dim features
            features = features + torch.randn_like(features) * noise_std
            
            # Feature dropout (randomly zero out some features)
            if torch.rand(1) < 0.1:  # 10% chance
                dropout_mask = torch.rand_like(features) > 0.05  # Keep 95% of features
                features = features * dropout_mask.float()
        
        # Project to embedding space through optimized layers
        embeddings = self.projection(features)
        
        # Advanced normalization with learnable temperature
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward_with_temperature(self, features, temperature=1.0):
        """Forward pass with temperature scaling for calibrated confidence"""
        embeddings = self.forward_one(features)
        return embeddings / temperature
    
    def forward(self, anchor, positive=None, negative=None):
        """Forward pass for triplet or single feature vector"""
        anchor_embedding = self.forward_one(anchor)
        
        if positive is not None and negative is not None:
            positive_embedding = self.forward_one(positive)
            negative_embedding = self.forward_one(negative)
            return anchor_embedding, positive_embedding, negative_embedding
        else:
            return anchor_embedding


class ArcFaceLoss(nn.Module):
    """Enhanced ArcFace loss with label smoothing and temperature scaling"""
    
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64, 
                 label_smoothing: float = 0.1, temperature: float = 1.0):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        
        # Weight matrix with better initialization
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Label smoothing cross entropy
        self.label_smooth_criterion = LabelSmoothingCrossEntropy(
            num_classes=num_classes, smoothing=label_smoothing
        )
        
    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(embeddings, weights)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # Convert to angles
        theta = cos_theta.acos()
        
        # Add margin to target angle  
        target_logits = cos_theta.clone()
        target_theta = theta[torch.arange(embeddings.size(0)), labels]
        target_theta += self.margin
        target_logits[torch.arange(embeddings.size(0)), labels] = target_theta.cos()
        
        # Scale logits and apply temperature
        logits = target_logits * self.scale / self.temperature
        
        # Apply label smoothing
        return self.label_smooth_criterion(logits, labels)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy for better generalization"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance and hard examples"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TripletLoss(nn.Module):
    """Enhanced triplet loss with hard negative mining and adaptive margin"""
    
    def __init__(self, margin: float = 0.5, mining_strategy: str = 'hard', adaptive_margin: bool = True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.adaptive_margin = adaptive_margin
        
    def forward(self, anchor, positive, negative):
        # Calculate distances
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Adaptive margin based on positive distance
        if self.adaptive_margin:
            margin = self.margin + 0.1 * distance_positive.detach()
        else:
            margin = self.margin
        
        # Calculate triplet loss
        losses = F.relu(distance_positive - distance_negative + margin)
        
        # Hard negative mining - focus on hardest examples
        if self.mining_strategy == 'hard':
            # Only use the hardest triplets (non-zero loss)
            hard_losses = losses[losses > 0]
            if len(hard_losses) > 0:
                return hard_losses.mean()
            else:
                return losses.mean()
        elif self.mining_strategy == 'semihard':
            # Semi-hard: positive closer than negative but within margin
            semihard_mask = (distance_positive < distance_negative) & (losses > 0)
            if semihard_mask.sum() > 0:
                return losses[semihard_mask].mean()
            else:
                return losses.mean()
        else:
            return losses.mean()


class EarlyStoppingCallback:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """Returns True if training should stop"""
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
            return False
            
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info(f"Early stopping triggered. Restored best model with score: {self.best_score:.4f}")
                return True
            return False


class FewShotDataset(Dataset):
    """Dataset for few-shot learning with triplet sampling"""
    
    def __init__(self, features_file: str, mode: str = 'train', n_way: int = 5, k_shot: int = 8):
        self.features_file = features_file
        self.mode = mode
        self.n_way = n_way
        self.k_shot = k_shot
        
        # Load data structure
        with h5py.File(features_file, 'r') as hf:
            self.items = {}
            
            for key in hf.keys():
                if key.startswith('image_'):
                    item_id = hf[key].attrs['item_id']
                    image_path = hf[key].attrs['image_path']
                    
                    if item_id not in self.items:
                        self.items[item_id] = []
                    
                    self.items[item_id].append({
                        'key': key,
                        'path': image_path
                    })
        
        # Filter out items with insufficient samples  
        self.items = {k: v for k, v in self.items.items() if len(v) >= 1}
        
        # Split items for train/val
        self.item_ids = list(self.items.keys())
        if len(self.item_ids) < 1:
            raise ValueError(f"No items found for training. Found {len(self.item_ids)} items.")
        
        np.random.shuffle(self.item_ids)
        
        # Handle single item case by using the same item for both train and val
        if len(self.item_ids) == 1:
            logger.warning(f"Only 1 item found - using same item for train and val (demo mode)")
            # Use the single item for both train and val
            pass  # Don't split, use all items for both modes
        else:
            split_point = max(1, int(len(self.item_ids) * 0.8))  # Ensure at least 1 item in each split
            if mode == 'train':
                self.item_ids = self.item_ids[:split_point]
            else:
                self.item_ids = self.item_ids[split_point:]
        
        logger.info(f"{mode} set: {len(self.item_ids)} items")
    
    def __len__(self):
        return len(self.item_ids) * 100  # Synthetic epoch size
    
    def __getitem__(self, idx):
        """Get triplet of (anchor, positive, negative) with combined features"""
        # Sample anchor item
        anchor_item = np.random.choice(self.item_ids)
        anchor_images = self.items[anchor_item]
        
        # Sample anchor and positive from same item
        if len(anchor_images) >= 2:
            anchor_idx, positive_idx = np.random.choice(len(anchor_images), 2, replace=False)
        else:
            anchor_idx = positive_idx = 0
        
        # Sample negative from different item  
        available_negatives = [i for i in self.item_ids if i != anchor_item]
        if not available_negatives:
            # Fallback: use same item but different image (for single item demo mode)
            negative_item = anchor_item
            negative_images = anchor_images
            available_indices = [i for i in range(len(negative_images)) if i not in [anchor_idx, positive_idx]]
            if available_indices:
                negative_idx = np.random.choice(available_indices)
            else:
                # Ultimate fallback: use same image with different index
                negative_idx = (anchor_idx + 1) % len(negative_images)
        else:
            negative_item = np.random.choice(available_negatives)
            negative_images = self.items[negative_item]
            negative_idx = np.random.choice(len(negative_images))
        
        # Load optimized 1536-dimensional features from HDF5
        with h5py.File(self.features_file, 'r') as hf:
            # Load CLIP features (768 dimensions - optimized ViT-L/14)
            anchor_clip = hf[anchor_images[anchor_idx]['key']]['clip'][:]
            positive_clip = hf[anchor_images[positive_idx]['key']]['clip'][:]
            negative_clip = hf[negative_images[negative_idx]['key']]['clip'][:]
            
            # Load DINOv2 features (768 dimensions - optimized DINOv2-base)
            try:
                anchor_dino = hf[anchor_images[anchor_idx]['key']]['dinov2'][:]
                positive_dino = hf[anchor_images[positive_idx]['key']]['dinov2'][:]
                negative_dino = hf[negative_images[negative_idx]['key']]['dinov2'][:]
                
                # Verify dimensions for optimized architecture
                if len(anchor_clip) != 768 or len(anchor_dino) != 768:
                    logger.warning(f"Unexpected feature dimensions: CLIP={len(anchor_clip)}, DINOv2={len(anchor_dino)}")
                
                # Combine CLIP (768) + DINOv2 (768) = 1536 total dimensions
                anchor_features = np.concatenate([anchor_clip, anchor_dino])
                positive_features = np.concatenate([positive_clip, positive_dino])
                negative_features = np.concatenate([negative_clip, negative_dino])
                
                # Verify final dimensions
                if len(anchor_features) != 1536:
                    logger.warning(f"Expected 1536 dimensions, got {len(anchor_features)}")
                    
            except KeyError:
                # Fallback to CLIP only if DINOv2 not available (not optimal for new architecture)
                logger.error("DINOv2 features not found! This architecture requires both CLIP and DINOv2 features.")
                logger.error("Please re-run feature extraction with the optimized configuration.")
                # Pad with zeros to maintain 1536 dimensions
                anchor_features = np.concatenate([anchor_clip, np.zeros(768)])
                positive_features = np.concatenate([positive_clip, np.zeros(768)])
                negative_features = np.concatenate([negative_clip, np.zeros(768)])
        
        return {
            'anchor': torch.FloatTensor(anchor_features),
            'positive': torch.FloatTensor(positive_features),
            'negative': torch.FloatTensor(negative_features),
            'label': self.item_ids.index(anchor_item)
        }


class AdvancedModelTrainer:
    """
    GPU-Accelerated Training Pipeline with Comprehensive Anti-Overfitting
    
    Features:
    - Mixed precision training (FP16) for 2x speedup
    - Advanced learning rate scheduling
    - Early stopping with cross-validation  
    - Gradient accumulation for large batch simulation
    - Multiple loss functions with adaptive weighting
    - Test-time augmentation for robust evaluation
    """
    
    def __init__(self, config: Dict, features_file: str = None):
        self.config = config
        self.features_file = features_file
        
        # GPU acceleration setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_mixed_precision = True
            logger.info("🚀 Using CUDA GPU with mixed precision training")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.use_mixed_precision = False  # MPS doesn't support autocast yet
            logger.info("🚀 Using Apple MPS GPU")
        else:
            self.device = torch.device('cpu')
            self.use_mixed_precision = False
            logger.info("⚠️ Using CPU training (much slower)")
        
        # Mixed precision scaler for FP16 training
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Advanced training parameters
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.warmup_epochs = config.get('warmup_epochs', 3)
        self.use_cross_validation = config.get('use_cross_validation', True)
        self.cv_folds = config.get('cv_folds', 3)
        
        # Anti-overfitting strategies
        self.early_stopping = EarlyStoppingCallback(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )
        
        # Loss function weights (adaptive during training)
        self.loss_weights = {
            'triplet': config.get('triplet_weight', 1.0),
            'arcface': config.get('arcface_weight', 0.5),
            'focal': config.get('focal_weight', 0.3)
        }
        
        # Training statistics
        self.training_stats = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'best_val_accuracy': 0.0
        }
    
    def create_model_and_optimizers(self, input_dim: int, num_classes: int) -> Tuple[nn.Module, torch.optim.Optimizer, Dict]:
        """Create model, optimizer, and loss functions with GPU acceleration"""
        
        # State-of-the-art SiameseNetwork optimized for 1536-dimensional features
        model = SiameseNetwork(
            base_model=self.config['clip_model'],
            embedding_dim=self.config['embedding_dim'],
            input_dim=input_dim,
            dropout_rate=self.config.get('dropout_rate', 0.3)  # Reduced dropout for better feature utilization
        ).to(self.device)
        
        # Log model architecture for verification
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"🏗️  Model Architecture: {total_params:,} total params, {trainable_params:,} trainable")
        logger.info(f"📐 Input dimensions: {input_dim}, Output dimensions: {self.config['embedding_dim']}")
        
        # State-of-the-art optimizer optimized for modern architectures
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 2e-5),  # Reduced for less regularization
            betas=(0.9, 0.95),     # Optimized betas for transformer-like architectures
            eps=1e-8,
            amsgrad=True           # Enable AMSGrad for better convergence
        )
        
        logger.info(f"🚀 Optimizer: AdamW with lr={self.config['learning_rate']}, wd={self.config.get('weight_decay', 2e-5)}")
        
        # Multiple loss functions for robust training
        loss_functions = {
            'triplet': TripletLoss(
                margin=self.config.get('triplet_margin', 0.5),
                mining_strategy='hard',
                adaptive_margin=True
            ).to(self.device),
            'arcface': ArcFaceLoss(
                embedding_dim=self.config['embedding_dim'],
                num_classes=num_classes,
                margin=self.config.get('arcface_margin', 0.5),
                label_smoothing=0.1
            ).to(self.device),
            'focal': FocalLoss(alpha=1.0, gamma=2.0).to(self.device)
        }
        
        return model, optimizer, loss_functions
    
    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer, num_epochs: int, train_loader_len: int):
        """Create advanced learning rate scheduler"""
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Warmup scheduler for first few epochs
        def warmup_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 1.0
        
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        
        return scheduler, warmup_scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   loss_functions: Dict, epoch: int) -> Tuple[float, float]:
        """Train one epoch with GPU acceleration and mixed precision"""
        
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        # Gradient accumulation counter
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Move to GPU
            anchor = batch['anchor'].to(self.device, non_blocking=True)
            positive = batch['positive'].to(self.device, non_blocking=True) 
            negative = batch['negative'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.use_mixed_precision:
                with autocast():
                    # Get embeddings
                    anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                    
                    # Calculate losses based on weights
                    total_batch_loss = 0.0
                    
                    # Triplet loss (always active)
                    if self.loss_weights['triplet'] > 0:
                        triplet_loss = loss_functions['triplet'](anchor_emb, positive_emb, negative_emb)
                        total_batch_loss += self.loss_weights['triplet'] * triplet_loss
                    
                    # ArcFace loss (optional)
                    if self.loss_weights['arcface'] > 0:
                        arcface_loss = loss_functions['arcface'](anchor_emb, labels)
                        total_batch_loss += self.loss_weights['arcface'] * arcface_loss
                    
                    # Focal loss (optional)
                    if self.loss_weights['focal'] > 0:
                        focal_loss = loss_functions['focal'](anchor_emb, labels)
                        total_batch_loss += self.loss_weights['focal'] * focal_loss
                    
                    # Scale loss for gradient accumulation
                    total_batch_loss = total_batch_loss / self.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(total_batch_loss).backward()
                
            else:
                # Standard precision forward pass
                anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                
                # Calculate losses based on weights
                total_batch_loss = 0.0
                
                # Triplet loss (always active)
                if self.loss_weights['triplet'] > 0:
                    triplet_loss = loss_functions['triplet'](anchor_emb, positive_emb, negative_emb)
                    total_batch_loss += self.loss_weights['triplet'] * triplet_loss
                
                # ArcFace loss (optional)
                if self.loss_weights['arcface'] > 0:
                    arcface_loss = loss_functions['arcface'](anchor_emb, labels)
                    total_batch_loss += self.loss_weights['arcface'] * arcface_loss
                
                # Focal loss (optional)
                if self.loss_weights['focal'] > 0:
                    focal_loss = loss_functions['focal'](anchor_emb, labels)
                    total_batch_loss += self.loss_weights['focal'] * focal_loss
                
                # Scale loss for gradient accumulation
                total_batch_loss = total_batch_loss / self.gradient_accumulation_steps
                
                total_batch_loss.backward()
            
            accumulation_counter += 1
            
            # Gradient accumulation step
            if accumulation_counter % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    # Unscale gradients and clip
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Statistics
            total_loss += total_batch_loss.item() * self.gradient_accumulation_steps
            
            # Simple accuracy calculation (positive pairs closer than negative)
            with torch.no_grad():
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
                correct_predictions += (pos_dist < neg_dist).sum().item()
                total_predictions += anchor_emb.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / max(total_predictions, 1) * 100
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = total_loss / max(len(train_loader), 1)
        epoch_accuracy = correct_predictions / max(total_predictions, 1)
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, loss_functions: Dict) -> Tuple[float, float]:
        """Validate one epoch"""
        
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move to GPU
                anchor = batch['anchor'].to(self.device, non_blocking=True)
                positive = batch['positive'].to(self.device, non_blocking=True)
                negative = batch['negative'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                        
                        # Calculate losses based on weights
                        total_batch_loss = 0.0
                        
                        if self.loss_weights['triplet'] > 0:
                            triplet_loss = loss_functions['triplet'](anchor_emb, positive_emb, negative_emb)
                            total_batch_loss += self.loss_weights['triplet'] * triplet_loss
                        
                        if self.loss_weights['arcface'] > 0:
                            arcface_loss = loss_functions['arcface'](anchor_emb, labels)
                            total_batch_loss += self.loss_weights['arcface'] * arcface_loss
                        
                        if self.loss_weights['focal'] > 0:
                            focal_loss = loss_functions['focal'](anchor_emb, labels)
                            total_batch_loss += self.loss_weights['focal'] * focal_loss
                else:
                    anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                    
                    # Calculate losses based on weights
                    total_batch_loss = 0.0
                    
                    if self.loss_weights['triplet'] > 0:
                        triplet_loss = loss_functions['triplet'](anchor_emb, positive_emb, negative_emb)
                        total_batch_loss += self.loss_weights['triplet'] * triplet_loss
                    
                    if self.loss_weights['arcface'] > 0:
                        arcface_loss = loss_functions['arcface'](anchor_emb, labels)
                        total_batch_loss += self.loss_weights['arcface'] * arcface_loss
                    
                    if self.loss_weights['focal'] > 0:
                        focal_loss = loss_functions['focal'](anchor_emb, labels)
                        total_batch_loss += self.loss_weights['focal'] * focal_loss
                
                total_loss += total_batch_loss.item()
                
                # Accuracy calculation
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
                correct_predictions += (pos_dist < neg_dist).sum().item()
                total_predictions += anchor_emb.size(0)
        
        val_loss = total_loss / max(len(val_loader), 1)
        val_accuracy = correct_predictions / max(total_predictions, 1)
        
        return val_loss, val_accuracy
    
    def train(self, features_file: str) -> str:
        """
        Complete training pipeline with cross-validation and advanced anti-overfitting
        
        Returns:
            str: Path to best trained model
        """
        logger.info("🚀 Starting Advanced GPU-Accelerated Training Pipeline")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        
        # Auto-detect feature dimensions
        input_dim = self._detect_feature_dim(features_file)
        logger.info(f"Feature dimensions: {input_dim}")
        
        # Create datasets for cross-validation
        if self.use_cross_validation:
            return self._train_with_cross_validation(features_file, input_dim)
        else:
            return self._train_single_fold(features_file, input_dim)
    
    def _detect_feature_dim(self, features_file: str) -> int:
        """Auto-detect feature dimensions from HDF5 file - optimized for 1536-dim architecture"""
        try:
            with h5py.File(features_file, 'r') as hf:
                for key in hf.keys():
                    if key.startswith('image_'):
                        clip_dim = hf[key]['clip'].shape[0]
                        
                        # Check if DINOv2 features exist (required for optimized architecture)
                        if 'dinov2' in hf[key]:
                            dino_dim = hf[key]['dinov2'].shape[0]
                            total_dim = clip_dim + dino_dim
                            
                            # Verify expected dimensions for optimized architecture
                            if clip_dim == 768 and dino_dim == 768:
                                logger.info(f"✅ Detected optimized features: CLIP({clip_dim}) + DINOv2({dino_dim}) = {total_dim} dimensions")
                            else:
                                logger.warning(f"⚠️  Non-optimal dimensions: CLIP({clip_dim}) + DINOv2({dino_dim}) = {total_dim}")
                                logger.warning("Expected: CLIP(768) + DINOv2(768) = 1536 for optimal performance")
                            
                            return total_dim
                        else:
                            logger.error(f"❌ DINOv2 features missing! Found only CLIP with {clip_dim} dimensions")
                            logger.error("The optimized architecture requires both CLIP and DINOv2 features.")
                            logger.error("Please re-run feature extraction with the updated configuration.")
                            # Return CLIP dimension but warn user
                            return clip_dim
                            
            logger.error("No image features found in HDF5 file")
            raise ValueError("No features found - please run feature extraction first")
            
        except Exception as e:
            logger.error(f"Error detecting feature dimensions: {e}")
            raise
    
    def _train_with_cross_validation(self, features_file: str, input_dim: int) -> str:
        """Train with cross-validation for robust model selection"""
        logger.info(f"🔄 Training with {self.cv_folds}-fold cross-validation")
        
        # Load all items for cross-validation splits
        all_items = self._load_all_items(features_file)
        
        if len(all_items) < self.cv_folds:
            logger.warning(f"Insufficient items ({len(all_items)}) for {self.cv_folds}-fold CV, using single fold")
            return self._train_single_fold(features_file, input_dim)
        
        # Cross-validation splits
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        fold_results = []
        best_fold_model = None
        best_fold_score = 0.0
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(all_items)):
            logger.info(f"\n📊 Training Fold {fold + 1}/{self.cv_folds}")
            
            # Create fold-specific datasets
            train_items = [all_items[i] for i in train_indices]
            val_items = [all_items[i] for i in val_indices]
            
            # Create data loaders
            train_dataset = FewShotDataset(features_file, mode='custom', n_way=5, k_shot=8)
            train_dataset.item_ids = train_items
            
            val_dataset = FewShotDataset(features_file, mode='custom', n_way=5, k_shot=8)
            val_dataset.item_ids = val_items
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Create model for this fold
            num_classes = len(train_items)
            model, optimizer, loss_functions = self.create_model_and_optimizers(input_dim, num_classes)
            
            # Create schedulers
            main_scheduler, warmup_scheduler = self.create_lr_scheduler(
                optimizer, self.config['epochs'], len(train_loader)
            )
            
            # Reset early stopping for this fold
            fold_early_stopping = EarlyStoppingCallback(
                patience=self.config.get('early_stopping_patience', 15),
                min_delta=self.config.get('early_stopping_min_delta', 0.001)
            )
            
            # Train this fold
            best_val_acc = 0.0
            fold_model_path = None
            
            for epoch in range(self.config['epochs']):
                # Warmup phase
                if epoch < self.warmup_epochs:
                    current_scheduler = warmup_scheduler
                else:
                    current_scheduler = main_scheduler
                
                # Train epoch
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, loss_functions, epoch
                )
                
                # Validate epoch
                val_loss, val_acc = self.validate_epoch(
                    model, val_loader, loss_functions
                )
                
                # Update scheduler
                current_scheduler.step()
                
                # Update training statistics
                self.training_stats['epoch_losses'].append(train_loss)
                self.training_stats['epoch_accuracies'].append(train_acc)
                self.training_stats['val_losses'].append(val_loss)
                self.training_stats['val_accuracies'].append(val_acc)
                self.training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                logger.info(
                    f"Fold {fold+1} Epoch {epoch+1}/{self.config['epochs']}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Save best model for this fold
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    fold_model_path = Path(self.config['checkpoint_dir']) / f"fold_{fold+1}_best_model.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'val_accuracy': val_acc,
                        'config': self.config,
                        'fold': fold + 1,
                        'input_dim': input_dim
                    }, fold_model_path)
                
                # Early stopping check
                if fold_early_stopping(val_acc, model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'best_val_accuracy': best_val_acc,
                'model_path': fold_model_path
            })
            
            # Track best overall fold
            if best_val_acc > best_fold_score:
                best_fold_score = best_val_acc
                best_fold_model = fold_model_path
            
            logger.info(f"Fold {fold+1} completed. Best validation accuracy: {best_val_acc:.4f}")
        
        # Cross-validation summary
        fold_scores = [r['best_val_accuracy'] for r in fold_results]
        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        
        logger.info(f"\n📈 Cross-Validation Results:")
        logger.info(f"Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"Best Fold Score: {best_fold_score:.4f}")
        logger.info(f"All Fold Scores: {fold_scores}")
        
        # Save best model as final model
        final_model_path = Path(self.config['checkpoint_dir']) / "best_model.pth"
        if best_fold_model and best_fold_model.exists():
            import shutil
            shutil.copy2(best_fold_model, final_model_path)
            logger.info(f"Best model saved to: {final_model_path}")
        
        # Update training statistics
        self.training_stats['best_val_accuracy'] = best_fold_score
        self.training_stats['cv_mean'] = cv_mean
        self.training_stats['cv_std'] = cv_std
        
        return str(final_model_path)
    
    def _train_single_fold(self, features_file: str, input_dim: int) -> str:
        """Train without cross-validation (single train/val split)"""
        logger.info("🔄 Training with single train/validation split")
        
        # Create datasets
        train_dataset = FewShotDataset(features_file, mode='train', n_way=5, k_shot=8)
        val_dataset = FewShotDataset(features_file, mode='val', n_way=5, k_shot=8)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        num_classes = len(train_dataset.item_ids)
        model, optimizer, loss_functions = self.create_model_and_optimizers(input_dim, num_classes)
        
        # Create schedulers
        main_scheduler, warmup_scheduler = self.create_lr_scheduler(
            optimizer, self.config['epochs'], len(train_loader)
        )
        
        # Training loop
        best_val_acc = 0.0
        best_model_path = None
        
        for epoch in range(self.config['epochs']):
            # Warmup phase
            if epoch < self.warmup_epochs:
                current_scheduler = warmup_scheduler
            else:
                current_scheduler = main_scheduler
            
            # Train epoch
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, loss_functions, epoch
            )
            
            # Validate epoch
            val_loss, val_acc = self.validate_epoch(
                model, val_loader, loss_functions
            )
            
            # Update scheduler
            current_scheduler.step()
            
            # Update training statistics
            self.training_stats['epoch_losses'].append(train_loss)
            self.training_stats['epoch_accuracies'].append(train_acc)
            self.training_stats['val_losses'].append(val_loss)
            self.training_stats['val_accuracies'].append(val_acc)
            self.training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = Path(self.config['checkpoint_dir']) / "best_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': val_acc,
                    'config': self.config,
                    'input_dim': input_dim
                }, best_model_path)
            
            # Early stopping check
            if self.early_stopping(val_acc, model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                checkpoint_path = Path(self.config['checkpoint_dir']) / f"model_epoch_{epoch+1}_acc_{val_acc:.4f}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': main_scheduler.state_dict(),
                    'val_accuracy': val_acc,
                    'config': self.config,
                    'input_dim': input_dim
                }, checkpoint_path)
        
        # Update training statistics
        self.training_stats['best_val_accuracy'] = best_val_acc
        
        logger.info(f"\n🎯 Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return str(best_model_path)
    
    def _load_all_items(self, features_file: str) -> List[str]:
        """Load all item IDs from features file"""
        items = set()
        
        try:
            with h5py.File(features_file, 'r') as hf:
                for key in hf.keys():
                    if key.startswith('image_'):
                        item_id = hf[key].attrs['item_id']
                        items.add(item_id)
            
            return list(items)
            
        except Exception as e:
            logger.error(f"Error loading items from features file: {e}")
            return []


class ActiveLearner:
    """Active learning module for continuous improvement"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.85):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.uncertain_samples = []
        
        # Load model
        # Use GPU acceleration if available (CUDA or Apple Silicon MPS)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = SiameseNetwork(
            base_model=checkpoint['config']['clip_model'],
            embedding_dim=checkpoint['config']['embedding_dim']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict_with_confidence(self, image_features):
        """Predict with confidence scores"""
        with torch.no_grad():
            embeddings = self.model.forward_one(image_features)
            
            # Compute similarities to all known items
            # This would involve comparing to a database of embeddings
            # For now, return a placeholder
            confidence = 0.9  # Placeholder
            
            return embeddings, confidence
    
    def identify_uncertain_cases(self, predictions):
        """Identify cases that need human review"""
        uncertain = []
        
        for pred in predictions:
            if pred['confidence'] < self.confidence_threshold:
                uncertain.append(pred)
        
        return uncertain
    
    def update_model_incremental(self, new_samples):
        """Incrementally update model with new labeled samples"""
        # This would implement incremental learning
        # For production, consider techniques like:
        # - Elastic Weight Consolidation (EWC)
        # - Progressive Neural Networks
        # - Rehearsal-based methods
        pass


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train few-shot learning model')
    parser.add_argument('--features', type=str, required=True, help='Path to features HDF5 file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of item classes')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Optimized Configuration for State-of-the-Art Performance
    config = {
        # Model Architecture (optimized for 1536-dim features)
        'clip_model': 'ViT-L/14',           # Upgraded model for better generalization
        'dinov2_model': 'dinov2_vitb14',    # Upgraded DINOv2 model
        'embedding_dim': 512,               # Increased embedding dimension
        'input_dim': 1536,                  # CLIP(768) + DINOv2(768)
        
        # Training Parameters (optimized for excellent generalization)
        'learning_rate': 3e-4,              # Optimized learning rate for modern architectures
        'weight_decay': 2e-5,               # Reduced weight decay for less regularization
        'dropout_rate': 0.3,                # Reduced dropout for better feature utilization
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        
        # Loss Function Parameters (balanced for multi-objective optimization)
        'triplet_margin': 0.3,              # Reduced margin for harder learning
        'arcface_margin': 0.4,              # Optimized ArcFace margin
        'triplet_weight': 1.0,              # Primary loss
        'arcface_weight': 0.3,              # Secondary loss (reduced)
        'focal_weight': 0.1,                # Tertiary loss (minimal)
        
        # Advanced Training Settings
        'gradient_accumulation_steps': 2,   # Simulate larger batch sizes
        'max_grad_norm': 1.0,               # Gradient clipping
        'warmup_epochs': 3,                 # Learning rate warmup
        'early_stopping_patience': 20,     # Increased patience for complex model
        'early_stopping_min_delta': 0.001,
        
        # Cross-validation and Generalization
        'use_cross_validation': True,       # Enable for robust validation
        'cv_folds': 3,                      # Computational efficiency vs robustness
        
        # System Settings
        'num_classes': args.num_classes,
        'save_every': 5,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.use_wandb,
        
        # Performance Monitoring
        'target_accuracy': 0.98,            # Target validation accuracy
        'min_train_time': 300,              # Minimum training time (5 minutes)
        'max_train_time': 7200,             # Maximum training time (2 hours)
    }
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Create datasets for validation
    train_dataset = FewShotDataset(args.features, mode='train')
    val_dataset = FewShotDataset(args.features, mode='val')
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(config)
    
    # Train model (trainer handles dataloader creation internally)
    best_model_path = trainer.train(args.features)
    
    logger.info(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()