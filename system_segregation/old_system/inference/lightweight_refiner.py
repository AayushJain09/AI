"""
Lightweight Refiner Model for Hybrid Recognition System

This module provides a simple lightweight neural network that refines
1536D raw features for complex recognition cases. It's designed to be
fast and memory-efficient while providing modest accuracy improvements.

DESIGN PHILOSOPHY:
- Lightweight: Fast inference for real-time recognition
- Simple architecture: 1536D → 512D → 256D refined features  
- Memory efficient: Minimal GPU/memory footprint
- Fallback ready: Graceful degradation if model unavailable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LightweightRefiner(nn.Module):
    """
    Lightweight refiner network for 1536D raw features.
    
    Simple feedforward architecture designed for fast inference
    and modest accuracy improvements on complex recognition cases.
    
    Architecture:
    - Input: 1536D raw features (CLIP + DINOv2)
    - Hidden: 512D with ReLU activation and dropout
    - Output: 256D refined features for enhanced matching
    """
    
    def __init__(self, 
                 input_dim: int = 1536,
                 hidden_dim: int = 512, 
                 output_dim: int = 256,
                 dropout_rate: float = 0.3):
        """
        Initialize lightweight refiner network.
        
        Args:
            input_dim: Input feature dimension (1536 for CLIP + DINOv2)
            hidden_dim: Hidden layer dimension  
            output_dim: Output refined feature dimension
            dropout_rate: Dropout rate for regularization
        """
        super(LightweightRefiner, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        
        # Simple feedforward architecture optimized for speed
        self.layers = nn.Sequential(
            # Input projection with batch normalization
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output projection with L2 normalization
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # Initialize weights for stable training
        self._initialize_weights()
        
        logger.info(f"LightweightRefiner initialized: {input_dim}D → {hidden_dim}D → {output_dim}D")
    
    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                # Standard initialization for batch norm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through refiner network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Refined features of shape (batch_size, output_dim)
        """
        # Apply feedforward layers
        refined = self.layers(x)
        
        # L2 normalize output features for cosine similarity
        refined = F.normalize(refined, p=2, dim=1)
        
        return refined
    
    def extract_features(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Extract refined features from raw feature vector.
        
        Convenience method for single feature vector refinement.
        
        Args:
            raw_features: Raw 1536D feature vector
            
        Returns:
            Refined 256D feature vector
        """
        if len(raw_features) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}D features, got {len(raw_features)}D")
        
        # Convert to tensor and add batch dimension
        x = torch.FloatTensor(raw_features).unsqueeze(0)
        
        # Move to same device as model
        x = x.to(next(self.parameters()).device)
        
        # Forward pass
        with torch.no_grad():
            refined = self.forward(x)
        
        # Return as numpy array
        return refined.squeeze(0).cpu().numpy()


def create_lightweight_refiner(device: Optional[torch.device] = None) -> LightweightRefiner:
    """
    Create and initialize a lightweight refiner model.
    
    Convenience function for creating a refiner with standard architecture.
    
    Args:
        device: Device to place model on (auto-detected if None)
        
    Returns:
        Initialized LightweightRefiner model
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps') 
        else:
            device = torch.device('cpu')
    
    model = LightweightRefiner(
        input_dim=1536,  # CLIP (768) + DINOv2 (768)
        hidden_dim=512,  # Reasonable hidden size
        output_dim=256,  # Compact refined features
        dropout_rate=0.3  # Moderate regularization
    )
    
    # Move to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Created lightweight refiner on {device}")
    return model


def save_refiner_checkpoint(model: LightweightRefiner, 
                           checkpoint_path: str,
                           metadata: Optional[dict] = None):
    """
    Save lightweight refiner model checkpoint.
    
    Args:
        model: Trained LightweightRefiner model
        checkpoint_path: Path to save checkpoint
        metadata: Optional training metadata
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'output_dim': model.output_dim,
        'model_class': 'LightweightRefiner'
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved refiner checkpoint: {checkpoint_path}")


def load_refiner_checkpoint(checkpoint_path: str, 
                           device: Optional[torch.device] = None) -> LightweightRefiner:
    """
    Load lightweight refiner model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Loaded LightweightRefiner model
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved architecture
    model = LightweightRefiner(
        input_dim=checkpoint.get('input_dim', 1536),
        hidden_dim=checkpoint.get('hidden_dim', 512),
        output_dim=checkpoint.get('output_dim', 256)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded refiner checkpoint: {checkpoint_path} on {device}")
    return model


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing LightweightRefiner...")
    
    # Create model
    refiner = create_lightweight_refiner()
    
    # Test with random features
    test_features = np.random.randn(1536).astype(np.float32)
    
    # Extract refined features
    refined = refiner.extract_features(test_features)
    
    print(f"✅ Test passed:")
    print(f"  Input: {test_features.shape} features")
    print(f"  Output: {refined.shape} refined features")
    print(f"  Output norm: {np.linalg.norm(refined):.3f}")
    
    # Test batch processing
    batch_features = torch.randn(4, 1536)
    batch_refined = refiner(batch_features)
    
    print(f"  Batch input: {batch_features.shape}")
    print(f"  Batch output: {batch_refined.shape}")
    
    print("🎉 LightweightRefiner test completed successfully!")