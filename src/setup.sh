#!/bin/bash
# Setup script for AI Recognition System

echo "Setting up AI Recognition System..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Create directory structure
mkdir -p data/{raw,augmented,processed,embeddings,models}
mkdir -p checkpoints
mkdir -p logs

# Download CLIP model
python -c "import clip; clip.load('ViT-B/32')"

echo "Setup complete!"
echo "To start using the system:"
echo "1. Place your images in data/raw/ITEM_ID/ (8 images per item)"
echo "2. Run: python main.py --step all"