"""
Lightweight Refinement Model for Hybrid Recognition System

A minimal neural network designed to refine raw CLIP+DINOv2 features for hard cases
where the baseline raw feature matching struggles. This model is intentionally
lightweight to maintain fast inference while providing improved discrimination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LightweightRefiner(nn.Module):
    """
    Lightweight refinement model for hard cases in hybrid recognition system.
    
    Architecture Philosophy:
    - Minimal 2-layer design for fast inference
    - Input: 1536D raw CLIP+DINOv2 features  
    - Output: 256D refined embeddings
    - Focus on improving discrimination for visually similar items
    """
    
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, 
                 output_dim: int = 256, dropout_rate: float = 0.1):
        super(LightweightRefiner, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Lightweight 2-layer architecture
        self.refiner = nn.Sequential(
            # Input projection with batch normalization
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output projection with normalization
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout_rate * 0.5)  # Lighter dropout for output
        )
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Normalized output tensor of shape (batch_size, output_dim)
        """
        # Pass through refiner network
        refined = self.refiner(x)
        
        # L2 normalize for cosine similarity matching
        normalized = F.normalize(refined, p=2, dim=1)
        
        return normalized
    
    def extract_features(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Extract refined features from raw features (inference mode)
        
        Args:
            raw_features: Raw CLIP+DINOv2 features (1536D)
            
        Returns:
            Refined features (256D)
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensor
            if isinstance(raw_features, np.ndarray):
                x = torch.FloatTensor(raw_features)
            else:
                x = raw_features
            
            # Ensure correct shape
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            
            # Move to device if model is on GPU
            device = next(self.parameters()).device
            x = x.to(device)
            
            # Extract refined features
            refined = self.forward(x)
            
            # Return as numpy array
            return refined.cpu().numpy()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training the lightweight refiner.
    Optimized for few-shot learning scenarios.
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings: Refined embeddings (batch_size, embedding_dim)
            labels: Item labels (batch_size,)
            
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute positive and negative similarities
        positive_sim = similarity_matrix * positive_mask
        negative_sim = similarity_matrix * negative_mask
        
        # Contrastive loss computation
        positive_loss = -torch.log(torch.exp(positive_sim).sum(dim=1) + 1e-8)
        negative_loss = torch.log(torch.exp(negative_sim).sum(dim=1) + 1e-8)
        
        # Combine losses
        loss = (positive_loss + negative_loss).mean()
        
        return loss


class LightweightRefinerTrainer:
    """
    Trainer for the lightweight refiner model.
    Designed for quick training on hard cases identified by the baseline system.
    """
    
    def __init__(self, model: LightweightRefiner, device: str = 'auto'):
        self.model = model
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training components
        self.criterion = ContrastiveLoss(margin=0.5, temperature=0.1)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        logger.info(f"LightweightRefinerTrainer initialized on {self.device}")
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            refined_embeddings = self.model(features)
            loss = self.criterion(refined_embeddings, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                refined_embeddings = self.model(features)
                loss = self.criterion(refined_embeddings, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'dropout_rate': self.model.dropout_rate
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str, device: str = 'auto') -> LightweightRefiner:
        """Load a trained model"""
        checkpoint = torch.load(path, map_location='cpu')
        
        config = checkpoint['model_config']
        model = LightweightRefiner(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path} on {device}")
        return model


def create_lightweight_refiner(config: Dict) -> LightweightRefiner:
    """
    Factory function to create a lightweight refiner from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized LightweightRefiner model
    """
    model_config = config.get('lightweight_model', {})
    
    return LightweightRefiner(
        input_dim=model_config.get('input_dim', 1536),
        hidden_dim=model_config.get('hidden_dim', 512),
        output_dim=model_config.get('output_dim', 256),
        dropout_rate=model_config.get('dropout_rate', 0.1)
    )


if __name__ == "__main__":
    # Test the model
    model = LightweightRefiner()
    
    # Test forward pass
    batch_size = 4
    input_dim = 1536
    test_input = torch.randn(batch_size, input_dim)
    
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {torch.norm(output, dim=1)}")  # Should be ~1.0 (normalized)
    
    logger.info("LightweightRefiner test passed!")