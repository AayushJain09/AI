# AI Recognition System Development Guide

## Project Overview
Build a high-accuracy (95%+) offline image recognition system for inventory management using only 8 images per item.

## System Architecture

```
ai-recognition-system/
├── data/
│   ├── raw/                    # Original 8 images per item
│   ├── augmented/             # Generated training data (400+ per item)
│   ├── processed/             # Preprocessed images
│   ├── embeddings/            # Extracted feature vectors
│   └── models/                # Trained models and indices
├── src/
│   ├── data_preparation/      # Image preprocessing & augmentation
│   ├── feature_extraction/    # CLIP & custom extractors
│   ├── training/              # Model training scripts
│   ├── inference/             # Recognition pipeline
│   └── evaluation/            # Testing & metrics
├── configs/                   # Configuration files
├── notebooks/                 # Jupyter notebooks for experiments
└── tests/                     # Unit and integration tests
```

## Development Phases

### Phase 1: Data Preparation (Week 1)
- Set up data collection protocol
- Implement augmentation pipeline
- Create synthetic data generator
- Build data validation tools

### Phase 2: Feature Engineering (Week 2)
- Implement multi-model feature extraction
- Create embedding storage system
- Build feature matching algorithms
- Develop similarity metrics

### Phase 3: Model Development (Week 3)
- Fine-tune CLIP model
- Implement Siamese networks
- Create ensemble system
- Build confidence scoring

### Phase 4: Optimization (Week 4)
- Implement caching system
- Optimize inference speed
- Add active learning
- Performance monitoring

## Key Technical Decisions (Updated Architecture)

1. **Core Models (Always Loaded)**:
   - **CLIP ViT-B/32**: Multi-modal vision-language features (512 dimensions)
   - **DINOv2-small**: Self-supervised visual features (384 dimensions)
2. **Additional Models (Full Mode Only)**:
   - **EfficientNet-B4**: CNN features for enhanced accuracy (1792 dimensions)
3. **Architecture Optimization**: ResNet removed (redundant with CLIP + DINOv2 combination)
4. **Mobile Mode Features**: CLIP + DINOv2 = 896 total dimensions (optimal speed/accuracy)
5. **Full Mode Features**: CLIP + DINOv2 + EfficientNet + Traditional CV features
6. **Augmentation Factor**: 50x per image (8 → 400 images)
7. **Similarity Metric**: Cosine similarity with threshold 0.85

## Dual-Mode Architecture

### Mobile Mode (Edge Deployment)
- **Models**: CLIP + DINOv2 only
- **Target Devices**: Microsoft Surface, tablets, edge devices
- **Performance**: ~800x faster than full mode
- **Use Case**: Real-time item addition, field inventory
- **Command**: `python main.py --mobile --step extract`

### Full Mode (Maximum Accuracy)
- **Models**: CLIP + DINOv2 + EfficientNet + Traditional CV
- **Target**: Desktop training and high-accuracy inference
- **Features**: Complete feature extraction pipeline
- **Use Case**: Training, batch processing, maximum accuracy needs
- **Command**: `python main.py --full --step extract`

### Reversible Switching
- Same codebase supports both modes
- Configuration-based switching via `mobile_mode` flag
- No separate implementations needed
- Instant mode switching with command-line flags

## Performance Targets

### Mobile Mode
- **Accuracy**: 92%+ on test set (CLIP + DINOv2 combination)
- **Inference Time**: <50ms per image
- **Memory Usage**: <2GB for 10,000 items
- **Feature Dimensions**: 896 (512 CLIP + 384 DINOv2)

### Full Mode  
- **Accuracy**: 95%+ on test set
- **Inference Time**: <500ms per image
- **Memory Usage**: <4GB for 10,000 items
- **Training Time**: <2 hours for 1,000 items

## Critical Success Factors

1. **Image Quality**: 4K resolution, consistent lighting
2. **Augmentation Diversity**: Cover all real-world variations
3. **Feature Robustness**: Multiple complementary extractors
4. **Verification Pipeline**: Multi-stage validation

## Development Rules

### For Terminal/CLI Operations
```bash
# Always activate virtual environment first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Use these aliases for common operations
alias prepare="python src/data_preparation/prepare.py"
alias train="python src/training/train.py"
alias test="python src/evaluation/test.py"
alias infer="python src/inference/recognize.py"

# Standard workflow
prepare --input data/raw --output data/augmented
train --config configs/training.yaml
test --model models/best_model.pth --data data/test
```

### For Cursor AI Editor
```javascript
// Cursor rules for this project
{
  "rules": [
    "Always use type hints in Python code",
    "Implement comprehensive error handling",
    "Add docstrings to all functions",
    "Use meaningful variable names",
    "Follow PEP 8 style guide",
    "Create unit tests for new functions",
    "Log all important operations",
    "Use configuration files, not hardcoded values"
  ],
  "codeSnippets": {
    "augmentation": "Use albumentations library with custom pipeline",
    "feature_extraction": "Implement both CLIP and traditional features",
    "similarity": "Use cosine similarity with normalization",
    "evaluation": "Calculate precision, recall, F1, and confusion matrix"
  }
}
```

## Quick Start Commands

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data
python prepare_data.py --input /path/to/images --items 100

# 3. Train model
python train_model.py --config configs/base.yaml

# 4. Evaluate
python evaluate.py --model models/latest.pth

# 5. Run inference
python recognize.py --image test.jpg --threshold 0.85
```

## Troubleshooting

### Low Accuracy (<90%)
1. Check image quality and resolution
2. Switch to full mode for maximum accuracy: `--full`
3. Increase augmentation diversity
4. Fine-tune similarity threshold
5. Verify both CLIP and DINOv2 are loading correctly

### Slow Inference (>1s)
1. Switch to mobile mode: `--mobile` (800x speedup)
2. Enable GPU acceleration (MPS for Apple Silicon)
3. Implement batch processing
4. Use model quantization
5. Optimize image preprocessing

### DINOv2 Loading Issues
1. Check internet connection for torch.hub download
2. Clear torch hub cache: `torch.hub.set_dir('/new/cache/path')`
3. Manually download model if needed
4. System falls back to CLIP-only if DINOv2 fails

### Memory Issues
1. Use incremental training
2. Implement embedding compression
3. Limit cache size
4. Use memory-mapped files

## Best Practices

1. **Version Control**: Track all experiments with MLflow/Weights & Biases
2. **Data Quality**: Validate all images before processing
3. **Model Checkpoints**: Save every 10 epochs
4. **Evaluation**: Test on held-out data regularly
5. **Documentation**: Update this file with learnings

## Contact & Support
- Project Lead: [Your Name]
- Technical Issues: Create GitHub issue
- Model Performance: Check evaluation metrics in `logs/`