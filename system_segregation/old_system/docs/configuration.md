# Configuration Guide - AI Recognition System

## Configuration Overview

The AI Recognition System uses a YAML configuration file (`config.yaml`) for all system settings. This guide explains all configuration options and their optimal values.

## Current Production Configuration

```yaml
# config.yaml - Current working configuration
confidence_threshold: 0.90
min_stage1_confidence: 0.7
model_path: "checkpoints/best_model_DISABLED.pth"

features:
  clip_model: "ViT-L/14"
  clip_dimensions: 768
  dinov2_dimensions: 768
  total_dimensions: 1536
  batch_size: 32

training:
  epochs: 40
  batch_size: 24
  learning_rate: 3e-4
  augmentations_per_image: 50

indexing:
  index_type: "Flat"
  metric: "cosine"
  normalize_features: true

model:
  use_siamese: false  # Disabled for stability
```

## Configuration Sections

### 1. Recognition Settings

#### Core Recognition Parameters
```yaml
# Confidence thresholds for recognition
confidence_threshold: 0.90        # Final confidence required for positive ID
min_stage1_confidence: 0.7        # Initial similarity threshold
max_candidates: 50                # Maximum candidates to consider

# Recognition behavior
enable_unknown_detection: true    # Enable rejection of unknown items
strict_mode: false               # Extra strict validation (slower)
```

**Parameter Details:**
- **confidence_threshold**: Higher = fewer false positives, may miss valid items
- **min_stage1_confidence**: Lower = more candidates, slower but more thorough
- **max_candidates**: Higher = more thorough search, slower performance

#### Optimal Values by Use Case
```yaml
# High accuracy mode (recommended)
confidence_threshold: 0.90
min_stage1_confidence: 0.7

# Fast mode (if speed is critical)
confidence_threshold: 0.80
min_stage1_confidence: 0.6
max_candidates: 20

# Ultra-strict mode (zero false positives)
confidence_threshold: 0.95
min_stage1_confidence: 0.8
strict_mode: true
```

### 2. Feature Extraction Settings

#### Model Configuration
```yaml
features:
  # Core models (always loaded)
  clip_model: "ViT-L/14"           # CLIP variant: ViT-B/32, ViT-L/14
  clip_dimensions: 768             # Output dimensions (auto-detected)
  dinov2_model: "dinov2_vitb14"    # DINOv2 variant
  dinov2_dimensions: 768           # Output dimensions (auto-detected)
  
  # Performance settings
  batch_size: 32                   # Images per batch
  device: "auto"                   # auto, cpu, cuda, mps
  normalize_features: true         # L2 normalization (recommended)
  
  # Combined settings
  total_dimensions: 1536           # clip_dims + dinov2_dims
  feature_type: "raw"              # raw features (no compression)
```

#### Device-Specific Optimization
```yaml
# Apple Silicon (M1/M2)
features:
  device: "mps"
  batch_size: 32

# NVIDIA GPU
features:
  device: "cuda"
  batch_size: 64    # Can handle larger batches

# CPU only
features:
  device: "cpu"
  batch_size: 8     # Smaller batches for CPU
```

### 3. Training Configuration

#### Basic Training Settings
```yaml
training:
  epochs: 40                       # Training iterations
  batch_size: 24                   # Samples per batch
  learning_rate: 3e-4              # Optimization step size
  
  # Data settings
  augmentations_per_image: 50      # Synthetic data multiplier
  validation_split: 0.2            # Fraction for validation
  
  # Optimization
  optimizer: "AdamW"               # Adam, AdamW, SGD
  scheduler: "CosineAnnealingLR"   # Learning rate schedule
  weight_decay: 1e-4               # Regularization
```

#### Advanced Training Options
```yaml
training:
  # Early stopping
  early_stopping:
    patience: 10                   # Epochs without improvement
    min_delta: 0.001              # Minimum improvement threshold
  
  # Model checkpointing
  save_every: 5                    # Save checkpoint every N epochs
  keep_best_only: true            # Only keep best performing model
  
  # Loss function
  loss_function: "ContrastiveLoss" # ContrastiveLoss, TripletLoss
  margin: 1.0                     # Loss margin parameter
  
  # Regularization
  dropout: 0.3                    # Dropout rate
  label_smoothing: 0.1            # Label smoothing factor
```

#### Training Presets
```yaml
# Quick development training
training_preset_fast:
  epochs: 10
  batch_size: 16
  learning_rate: 1e-3
  augmentations_per_image: 25

# Production training (recommended)
training_preset_production:
  epochs: 40
  batch_size: 24
  learning_rate: 3e-4
  augmentations_per_image: 50

# Maximum accuracy training
training_preset_max_accuracy:
  epochs: 60
  batch_size: 16
  learning_rate: 1e-4
  augmentations_per_image: 75
```

### 4. Data Augmentation Settings

#### Augmentation Pipeline
```yaml
augmentation:
  augmentations_per_image: 50      # Total augmented images per original
  
  # Geometric transformations
  geometric:
    rotation_limit: 45             # Degrees
    scale_limit: 0.2              # Scale factor range
    shift_limit: 0.1              # Translation range
    horizontal_flip: 0.5          # Probability
    
  # Color/lighting
  photometric:
    brightness_limit: 0.3         # Brightness adjustment range
    contrast_limit: 0.3           # Contrast adjustment range
    hue_shift_limit: 20           # Hue adjustment (degrees)
    saturation_limit: 30          # Saturation adjustment
    
  # Quality/noise
  degradation:
    noise_limit: 50               # Gaussian noise variance
    blur_limit: 3                 # Gaussian blur kernel size
    jpeg_quality: [70, 100]       # JPEG compression range
    
  # Advanced augmentations
  advanced:
    cutout_probability: 0.3       # Random patches removal
    mixup_alpha: 0.2             # Mixup augmentation strength
    background_replacement: 0.1   # Probability of background change
```

#### Augmentation Presets
```yaml
# Minimal augmentation (fast training)
augmentation_preset_minimal:
  augmentations_per_image: 25
  geometric_weight: 0.7
  photometric_weight: 0.3

# Balanced augmentation (recommended)
augmentation_preset_balanced:
  augmentations_per_image: 50
  geometric_weight: 0.5
  photometric_weight: 0.4
  degradation_weight: 0.1

# Aggressive augmentation (maximum robustness)
augmentation_preset_aggressive:
  augmentations_per_image: 75
  geometric_weight: 0.4
  photometric_weight: 0.3
  degradation_weight: 0.2
  advanced_weight: 0.1
```

### 5. Indexing Configuration

#### FAISS Index Settings
```yaml
indexing:
  # Index type selection
  index_type: "auto"               # auto, Flat, IVF, HNSW, PQ
  
  # Distance metric
  metric: "cosine"                 # cosine, euclidean, inner_product
  
  # Performance options
  use_gpu: true                    # GPU acceleration (if available)
  normalize_features: true         # Feature normalization
  
  # Auto-selection thresholds
  auto_selection:
    flat_threshold: 10000          # Use Flat index below this size
    ivf_threshold: 100000          # Use IVF index below this size
    # Use HNSW for larger indices
```

#### Index Type Details
```yaml
# Exact search (best accuracy, slower for large datasets)
index_config_exact:
  index_type: "Flat"
  metric: "cosine"
  
# Approximate search (good balance)
index_config_ivf:
  index_type: "IVF"
  n_clusters: 256                  # Number of clusters
  n_probe: 32                      # Clusters to search
  
# Fast approximate search
index_config_hnsw:
  index_type: "HNSW"
  M: 16                           # Connections per node
  ef_construction: 200            # Construction parameter
  ef_search: 64                   # Search parameter
  
# Compressed index (memory efficient)
index_config_pq:
  index_type: "IVF_PQ"
  n_clusters: 256
  pq_segments: 8                  # PQ segments
  pq_bits: 8                      # Bits per segment
```

### 6. System Configuration

#### Logging Settings
```yaml
logging:
  level: "INFO"                    # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # File logging
  file_logging:
    enabled: true
    log_file: "logs/system.log"
    max_size: "100MB"
    backup_count: 5
    
  # Console logging
  console_logging:
    enabled: true
    colored: true
```

#### API Configuration
```yaml
api:
  host: "0.0.0.0"                 # Bind address
  port: 8000                      # Port number
  workers: 1                      # Number of worker processes
  
  # Security
  cors_origins: ["*"]             # Allowed origins (restrict in production)
  max_file_size: "10MB"           # Maximum upload size
  
  # Performance
  timeout: 30                     # Request timeout (seconds)
  keep_alive: 75                  # Keep-alive timeout
```

#### GUI Configuration
```yaml
gui:
  # Window settings
  window_size: [1200, 800]        # Default window size
  resizable: true                 # Allow window resizing
  
  # Theme
  theme: "light"                  # light, dark, auto
  font_size: 10                   # Base font size
  
  # Performance
  update_interval: 1000           # GUI update interval (ms)
  max_log_lines: 1000            # Maximum log lines to display
```

## Environment-Specific Configurations

### Development Configuration
```yaml
# config_development.yaml
extends: "config.yaml"

# Override for development
training:
  epochs: 10
  batch_size: 8
  augmentations_per_image: 25

logging:
  level: "DEBUG"
  
features:
  batch_size: 16
```

### Production Configuration
```yaml
# config_production.yaml
extends: "config.yaml"

# Production optimizations
training:
  epochs: 40
  batch_size: 24
  augmentations_per_image: 50

logging:
  level: "INFO"
  console_logging:
    enabled: false  # Disable console logging in production

api:
  cors_origins: ["https://yourdomain.com"]  # Restrict origins
```

### Testing Configuration
```yaml
# config_testing.yaml
extends: "config.yaml"

# Fast testing configuration
training:
  epochs: 5
  batch_size: 4
  augmentations_per_image: 10

confidence_threshold: 0.8  # More lenient for testing
```

## Configuration Validation

### Automatic Validation
The system automatically validates configuration on startup:

```python
# Example validation rules
validation_rules = {
    "confidence_threshold": {"min": 0.0, "max": 1.0},
    "min_stage1_confidence": {"min": 0.0, "max": 1.0},
    "training.epochs": {"min": 1, "max": 1000},
    "training.batch_size": {"min": 1, "max": 512},
    "features.total_dimensions": {"min": 1, "max": 10000}
}
```

### Manual Validation
```bash
# Validate configuration file
python3 -c "
import yaml
from src.config.validator import validate_config

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
is_valid, errors = validate_config(config)
if is_valid:
    print('✅ Configuration is valid')
else:
    print('❌ Configuration errors:')
    for error in errors:
        print(f'  - {error}')
"
```

## Performance Tuning

### Memory Optimization
```yaml
# For systems with limited memory
memory_optimized:
  features:
    batch_size: 8
  training:
    batch_size: 8
  indexing:
    use_gpu: false
```

### Speed Optimization
```yaml
# For maximum speed
speed_optimized:
  confidence_threshold: 0.8
  min_stage1_confidence: 0.6
  max_candidates: 20
  features:
    batch_size: 64
  indexing:
    index_type: "HNSW"
```

### Accuracy Optimization
```yaml
# For maximum accuracy
accuracy_optimized:
  confidence_threshold: 0.95
  min_stage1_confidence: 0.8
  strict_mode: true
  training:
    epochs: 60
    augmentations_per_image: 75
  indexing:
    index_type: "Flat"
```

## Configuration Management

### Loading Configurations
```python
# Load specific configuration
python3 your_script.py --config config_production.yaml

# Override specific values
python3 your_script.py --config config.yaml --override training.epochs=20
```

### Environment Variables
```bash
# Override via environment variables
export AI_RECOGNITION_CONFIDENCE_THRESHOLD=0.95
export AI_RECOGNITION_BATCH_SIZE=16

# Use in configuration
confidence_threshold: ${AI_RECOGNITION_CONFIDENCE_THRESHOLD:0.90}
```

This configuration system provides flexible, environment-specific settings while maintaining reasonable defaults for all use cases.