import os
import io
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import clip
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from dataclasses import dataclass
from pathlib import Path
import faiss
import pickle

# Configuration
@dataclass
class Config:
    model_name: str = "ViT-B/32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim: int = 512
    similarity_threshold: float = 0.85
    top_k: int = 5
    
    # Paths
    model_path: Path = Path("../data/models")
    embeddings_path: Path = Path("../data/models/embeddings")
    index_path: Path = Path("../data/models/faiss_index.bin")
    metadata_path: Path = Path("../data/models/metadata.json")

config = Config()

# Initialize FastAPI app
app = FastAPI(title="Inventory Recognition Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePreprocessor:
    """Advanced image preprocessing for better recognition accuracy"""
    
    def __init__(self):
        self.clip_preprocess = None
        self.target_size = (224, 224)
        
    def set_clip_preprocess(self, preprocess):
        self.clip_preprocess = preprocess
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply preprocessing pipeline"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply CLIP preprocessing
        if self.clip_preprocess:
            return self.clip_preprocess(image)
        
        # Fallback preprocessing
        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        return transform(image)
    
    def enhance_image(self, image_array: np.ndarray) -> np.ndarray:
        """Enhance image quality for better recognition"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced

class RecognitionModel:
    """Main recognition model with ensemble approach"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(
            config.model_name, 
            device=self.device
        )
        self.clip_model.eval()
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        self.preprocessor.set_clip_preprocess(self.clip_preprocess)
        
        # Load or create FAISS index
        self.index = None
        self.metadata = {}
        self.load_or_create_index()
        
    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if self.config.index_path.exists():
            self.index = faiss.read_index(str(self.config.index_path))
            
            if self.config.metadata_path.exists():
                with open(self.config.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.config.embedding_dim)  # Inner product
            self.metadata = {"items": {}, "index_to_item": {}}
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP features from image"""
        # Preprocess image
        image_tensor = self.preprocessor.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=1, keepdim=True)  # Normalize
            
        return features.cpu().numpy().flatten()
    
    def add_item(self, item_id: str, images: List[Image.Image]) -> Dict[str, Any]:
        """Add new item with multiple images to index"""
        embeddings = []
        
        for img in images:
            # Extract features
            features = self.extract_features(img)
            embeddings.append(features)
            
            # Add to FAISS index
            self.index.add(features.reshape(1, -1))
            
            # Update metadata
            idx = self.index.ntotal - 1
            self.metadata["index_to_item"][str(idx)] = item_id
        
        # Store item metadata
        if item_id not in self.metadata["items"]:
            self.metadata["items"][item_id] = {
                "indices": [],
                "added_at": datetime.now().isoformat()
            }
        
        start_idx = self.index.ntotal - len(images)
        self.metadata["items"][item_id]["indices"].extend(
            list(range(start_idx, self.index.ntotal))
        )
        
        # Save index and metadata
        self.save_index()
        
        return {
            "item_id": item_id,
            "num_embeddings": len(embeddings),
            "status": "success"
        }
    
    def search(self, query_image: Image.Image, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar items"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Extract features from query image
        query_features = self.extract_features(query_image)
        query_features = query_features.reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_features, top_k * 3)  # Get more to aggregate
        
        # Aggregate results by item
        item_scores = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
                
            item_id = self.metadata["index_to_item"].get(str(idx))
            if item_id:
                if item_id not in item_scores:
                    item_scores[item_id] = []
                item_scores[item_id].append(float(dist))
        
        # Calculate final scores (max score per item)
        results = []
        for item_id, scores in item_scores.items():
            max_score = max(scores)
            if max_score >= self.config.similarity_threshold:
                results.append({
                    "item_id": item_id,
                    "confidence": float(max_score),
                    "num_matches": len(scores)
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results[:top_k]
    
    def save_index(self):
        """Save FAISS index and metadata"""
        # Create directory if needed
        self.config.model_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.config.index_path))
        
        # Save metadata
        with open(self.config.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

# Initialize model
model = RecognitionModel(config)

@app.get("/")
async def root():
    return {"message": "Inventory Recognition Service", "status": "running"}

@app.post("/recognize")
async def recognize_image(file: UploadFile = File(...)):
    """Recognize item from uploaded image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Search for similar items
        results = model.search(image)
        
        if not results:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": "No matching items found",
                    "results": []
                }
            )
        
        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_item")
async def add_item(item_id: str, files: List[UploadFile] = File(...)):
    """Add new item with images to recognition database"""
    try:
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            images.append(image)
        
        result = model.add_item(item_id, images)
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_embeddings")
async def update_embeddings(item_id: str, files: List[UploadFile] = File(...)):
    """Update embeddings for existing item"""
    try:
        # Remove old embeddings
        if item_id in model.metadata["items"]:
            # Note: FAISS doesn't support removal, so we'll mark as inactive
            old_indices = model.metadata["items"][item_id]["indices"]
            for idx in old_indices:
                del model.metadata["index_to_item"][str(idx)]
        
        # Add new embeddings
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            images.append(image)
        
        result = model.add_item(item_id, images)
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get recognition system statistics"""
    return {
        "total_items": len(model.metadata["items"]),
        "total_embeddings": model.index.ntotal if model.index else 0,
        "model": config.model_name,
        "device": config.device,
        "similarity_threshold": config.similarity_threshold
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)