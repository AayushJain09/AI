#!/usr/bin/env python3
"""
Create placeholder lightweight refiner checkpoint.

This script creates a simple placeholder refiner model checkpoint
that can be used for testing the unified recognition pipeline.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inference.lightweight_refiner import LightweightRefiner, save_refiner_checkpoint


def create_placeholder_refiner():
    """Create and save a placeholder lightweight refiner."""
    print("🔧 Creating placeholder lightweight refiner...")
    
    # Create model with standard architecture
    model = LightweightRefiner(
        input_dim=1536,  # CLIP (768) + DINOv2 (768)
        hidden_dim=512,  # Hidden layer size
        output_dim=256,  # Refined feature size
        dropout_rate=0.3
    )
    
    # Initialize with random weights (placeholder)
    # In production, this would be a trained model
    print("⚠️  Note: This is a placeholder with random weights")
    print("   For production use, train the refiner on real data")
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "lightweight_refiner.pth"
    
    metadata = {
        'model_type': 'placeholder',
        'training_status': 'untrained',
        'note': 'Placeholder model for testing unified recognition pipeline'
    }
    
    save_refiner_checkpoint(model, str(checkpoint_path), metadata)
    
    print(f"✅ Placeholder refiner saved: {checkpoint_path}")
    print(f"📊 Architecture: {model.input_dim}D → {model.hidden_dim}D → {model.output_dim}D")
    
    # Test the saved model
    print("\n🧪 Testing saved model...")
    
    from inference.lightweight_refiner import load_refiner_checkpoint
    
    # Load the saved model
    loaded_model = load_refiner_checkpoint(str(checkpoint_path))
    
    # Test with random features
    test_features = np.random.randn(1536).astype(np.float32)
    refined_features = loaded_model.extract_features(test_features)
    
    print(f"✅ Test passed:")
    print(f"  Input shape: {test_features.shape}")
    print(f"  Output shape: {refined_features.shape}")
    print(f"  Output norm: {np.linalg.norm(refined_features):.3f}")
    
    return checkpoint_path


if __name__ == "__main__":
    try:
        checkpoint_path = create_placeholder_refiner()
        print(f"\n🎉 Placeholder refiner created successfully!")
        print(f"📁 Location: {checkpoint_path}")
        print(f"🔄 Ready for unified recognition pipeline testing")
        
    except Exception as e:
        print(f"\n❌ Failed to create placeholder refiner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)