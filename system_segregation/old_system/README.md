# OLD SYSTEM FILES (Original Proven Approach)

This folder contains all files related to your original proven system that achieved 99%+ accuracy.

## 📁 STRUCTURE

### Core Processing Files
- `src/data_preparation/prepare.py` - Your original proven augmentation pipeline
- `src/inference/recognize.py` - Original recognition system
- `src/feature_extraction/feature_extractor.py` - Original CLIP + DINOv2 extractor
- `src/indexing/faiss_indexer.py` - Original FAISS indexing system
- `main.py` - Original system entry point

### Data Storage (Old Format)
- `data/augmented/` - Stored augmented images (JPG files)
- `data/features.h5` - Feature vectors in HDF5 format
- `data/models/faiss_index.bin` - FAISS vector index
- `data/models/index_metadata.pkl` - FAISS metadata
- `data/models/index_stats.json` - FAISS performance stats

### Original Frontend/Backend
- `frontend/` - Original GUI system
- `backend/` - Original API system

## 🎯 PROVEN CONFIGURATION

Your original system used these exact settings for 99%+ accuracy:

**Strategy Weights:**
- Geometric: 30% (rotation, flip, scale)
- Perspective: 25% (3D transforms)
- Lighting: 25% (brightness, contrast)
- Noise/Blur: 15% (sensor variations)
- Effects: 5% (environmental)

**Processing:**
- 30+ augmentations per image
- Background removal with rembg
- CLIP + DINOv2 feature extraction (1536D)
- FAISS similarity search

## 📊 CHARACTERISTICS
- **Storage**: ~5GB for augmented images
- **Format**: Files scattered across directories
- **Search**: FAISS binary index
- **Processing**: Batch file generation