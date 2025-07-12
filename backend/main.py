#!/usr/bin/env python3
"""
AI Recognition System - Backend API
FastAPI-based REST API for the AI recognition system
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import logging
import json
import tempfile
import shutil
from pathlib import Path
import time
from datetime import datetime
import base64

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our AI system components
from main import AIRecognitionSystem
from src.inference.recognize import create_pipeline, RecognitionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Recognition System API",
    description="Backend API for inventory recognition system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
ai_system: Optional[AIRecognitionSystem] = None
recognition_pipeline = None
system_status = {
    "initialized": False,
    "training_in_progress": False,
    "evaluation_in_progress": False,
    "last_error": None,
    "system_ready": False
}

# Background task tracking
background_tasks = {}

# Pydantic models
class SystemStatus(BaseModel):
    initialized: bool
    training_in_progress: bool
    evaluation_in_progress: bool
    last_error: Optional[str]
    system_ready: bool

class ItemCreate(BaseModel):
    item_id: str
    name: str
    description: Optional[str] = ""
    category: Optional[str] = ""

class RecognitionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    confidence_threshold: Optional[float] = 0.85

class TrainingConfig(BaseModel):
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 16
    learning_rate: Optional[float] = 0.0001

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the AI system on startup"""
    global ai_system, recognition_pipeline, system_status
    
    try:
        logger.info("🚀 Starting AI Recognition System Backend...")
        
        # Create necessary directories
        os.makedirs("backend/logs", exist_ok=True)
        os.makedirs("backend/temp", exist_ok=True)
        os.makedirs("backend/uploads", exist_ok=True)
        
        # Initialize AI system
        config_path = "config.yaml"
        if Path(config_path).exists():
            ai_system = AIRecognitionSystem(config_path)
            logger.info("✅ AI System initialized successfully")
            
            # Try to load recognition pipeline
            try:
                recognition_pipeline = create_pipeline(config_path)
                system_status["system_ready"] = True
                logger.info("✅ Recognition pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  Recognition pipeline not ready: {e}")
                system_status["system_ready"] = False
            
            system_status["initialized"] = True
            
        else:
            logger.error("❌ Configuration file not found")
            system_status["last_error"] = "Configuration file not found"
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        system_status["last_error"] = str(e)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check system health status"""
    return {
        "status": "healthy" if system_status["initialized"] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "system_status": system_status
    }

# System status endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status"""
    return SystemStatus(**system_status)

# Initialize system endpoint
@app.post("/api/system/initialize")
async def initialize_system():
    """Initialize or reinitialize the AI system"""
    global ai_system, recognition_pipeline, system_status
    
    try:
        if not Path("config.yaml").exists():
            raise HTTPException(status_code=400, detail="Configuration file not found")
        
        ai_system = AIRecognitionSystem("config.yaml")
        recognition_pipeline = create_pipeline("config.yaml")
        
        system_status.update({
            "initialized": True,
            "system_ready": True,
            "last_error": None
        })
        
        logger.info("✅ System reinitialized successfully")
        return ApiResponse(success=True, message="System initialized successfully")
        
    except Exception as e:
        error_msg = f"Failed to initialize system: {str(e)}"
        logger.error(f"❌ {error_msg}")
        system_status["last_error"] = error_msg
        raise HTTPException(status_code=500, detail=error_msg)

# Item management endpoints
@app.get("/api/items")
async def get_items():
    """Get list of all items in the system"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        raw_dir = Path(ai_system.config['data']['raw_images_dir'])
        items = []
        
        if raw_dir.exists():
            for item_dir in raw_dir.iterdir():
                if item_dir.is_dir():
                    image_files = list(item_dir.glob('*.jpg')) + list(item_dir.glob('*.JPG')) + \
                                 list(item_dir.glob('*.png')) + list(item_dir.glob('*.PNG'))
                    
                    items.append({
                        "item_id": item_dir.name,
                        "name": item_dir.name,
                        "image_count": len(image_files),
                        "images": [str(f.name) for f in image_files[:5]]  # First 5 images
                    })
        
        return ApiResponse(success=True, message="Items retrieved successfully", data=items)
        
    except Exception as e:
        error_msg = f"Failed to get items: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/items")
async def create_item(item: ItemCreate):
    """Create a new item in the system"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        raw_dir = Path(ai_system.config['data']['raw_images_dir'])
        item_dir = raw_dir / item.item_id
        
        if item_dir.exists():
            raise HTTPException(status_code=400, detail="Item already exists")
        
        item_dir.mkdir(parents=True, exist_ok=True)
        
        # Create item metadata file
        metadata = {
            "item_id": item.item_id,
            "name": item.name,
            "description": item.description,
            "category": item.category,
            "created_at": datetime.now().isoformat()
        }
        
        with open(item_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Created new item: {item.item_id}")
        return ApiResponse(success=True, message=f"Item '{item.item_id}' created successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to create item: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/items/{item_id}/images")
async def upload_item_images(item_id: str, files: List[UploadFile] = File(...)):
    """Upload images for an item"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        raw_dir = Path(ai_system.config['data']['raw_images_dir'])
        item_dir = raw_dir / item_id
        
        if not item_dir.exists():
            raise HTTPException(status_code=404, detail="Item not found")
        
        uploaded_files = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            filename = f"{timestamp}_{file.filename}"
            file_path = item_dir / filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(filename)
        
        logger.info(f"✅ Uploaded {len(uploaded_files)} images for item {item_id}")
        return ApiResponse(
            success=True, 
            message=f"Uploaded {len(uploaded_files)} images successfully",
            data={"uploaded_files": uploaded_files}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to upload images: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.delete("/api/items/{item_id}")
async def delete_item(item_id: str):
    """Delete an item from the system"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        raw_dir = Path(ai_system.config['data']['raw_images_dir'])
        item_dir = raw_dir / item_id
        
        if not item_dir.exists():
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Remove directory and all contents
        shutil.rmtree(item_dir)
        
        logger.info(f"✅ Deleted item: {item_id}")
        return ApiResponse(success=True, message=f"Item '{item_id}' deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to delete item: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# Recognition endpoints
@app.post("/api/recognize")
async def recognize_image(request: RecognitionRequest):
    """Recognize an image and return the prediction"""
    try:
        if not recognition_pipeline:
            raise HTTPException(status_code=503, detail="Recognition pipeline not ready")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        try:
            # Run recognition
            start_time = time.time()
            result = recognition_pipeline.recognize(temp_path)
            processing_time = time.time() - start_time
            
            # Convert result to dict
            result_data = {
                "item_id": result.item_id,
                "confidence": result.confidence,
                "inference_time": result.inference_time,
                "processing_time": processing_time,
                "top_matches": result.top_k_matches[:5],
                "stage_results": result.stage_results
            }
            
            logger.info(f"🔍 Recognition result: {result.item_id} (confidence: {result.confidence:.3f})")
            return ApiResponse(
                success=True,
                message="Image recognized successfully",
                data=result_data
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Recognition failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/recognize/file")
async def recognize_uploaded_file(file: UploadFile = File(...)):
    """Recognize an uploaded image file"""
    try:
        if not recognition_pipeline:
            raise HTTPException(status_code=503, detail="Recognition pipeline not ready")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Run recognition
            result = recognition_pipeline.recognize(temp_path)
            
            result_data = {
                "item_id": result.item_id,
                "confidence": result.confidence,
                "inference_time": result.inference_time,
                "top_matches": result.top_k_matches[:5]
            }
            
            logger.info(f"🔍 File recognition result: {result.item_id} (confidence: {result.confidence:.3f})")
            return ApiResponse(
                success=True,
                message="Image recognized successfully",
                data=result_data
            )
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"File recognition failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# Training endpoints
@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks, config: TrainingConfig = None):
    """Start model training in the background"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        if system_status["training_in_progress"]:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        # Start training in background
        task_id = f"training_{int(time.time())}"
        background_tasks.add_task(run_training_pipeline, task_id, config)
        background_tasks[task_id] = {"status": "started", "progress": 0}
        
        system_status["training_in_progress"] = True
        
        logger.info(f"🚀 Started training task: {task_id}")
        return ApiResponse(
            success=True,
            message="Training started successfully",
            data={"task_id": task_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start training: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/train/status")
async def get_training_status():
    """Get current training status"""
    return ApiResponse(
        success=True,
        message="Training status retrieved",
        data={
            "training_in_progress": system_status["training_in_progress"],
            "tasks": background_tasks
        }
    )

# Evaluation endpoints
@app.post("/api/evaluate")
async def start_evaluation(background_tasks: BackgroundTasks):
    """Start system evaluation in the background"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        if system_status["evaluation_in_progress"]:
            raise HTTPException(status_code=400, detail="Evaluation already in progress")
        
        task_id = f"evaluation_{int(time.time())}"
        background_tasks.add_task(run_evaluation_pipeline, task_id)
        background_tasks[task_id] = {"status": "started", "progress": 0}
        
        system_status["evaluation_in_progress"] = True
        
        logger.info(f"🧪 Started evaluation task: {task_id}")
        return ApiResponse(
            success=True,
            message="Evaluation started successfully",
            data={"task_id": task_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start evaluation: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/evaluate/results")
async def get_evaluation_results():
    """Get latest evaluation results"""
    try:
        results_file = "evaluation_results.json"
        if not Path(results_file).exists():
            raise HTTPException(status_code=404, detail="No evaluation results found")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return ApiResponse(
            success=True,
            message="Evaluation results retrieved",
            data=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get evaluation results: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# Configuration endpoints
@app.get("/api/config")
async def get_configuration():
    """Get current system configuration"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        return ApiResponse(
            success=True,
            message="Configuration retrieved",
            data=ai_system.config
        )
        
    except Exception as e:
        error_msg = f"Failed to get configuration: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# Background task functions
async def run_training_pipeline(task_id: str, config: Optional[TrainingConfig]):
    """Run the complete training pipeline"""
    try:
        logger.info(f"🏃 Running training pipeline for task {task_id}")
        
        # Update task status
        background_tasks[task_id] = {"status": "preparing", "progress": 10}
        
        # Run data preparation
        if not ai_system.run_data_preparation():
            raise Exception("Data preparation failed")
        background_tasks[task_id] = {"status": "data_prepared", "progress": 30}
        
        # Run feature extraction
        if not ai_system.run_feature_extraction():
            raise Exception("Feature extraction failed")
        background_tasks[task_id] = {"status": "features_extracted", "progress": 50}
        
        # Run training
        if not ai_system.run_training():
            raise Exception("Model training failed")
        background_tasks[task_id] = {"status": "training_complete", "progress": 80}
        
        # Build recognition index
        if not ai_system.build_recognition_index():
            raise Exception("Index building failed")
        background_tasks[task_id] = {"status": "completed", "progress": 100}
        
        # Reload recognition pipeline
        global recognition_pipeline
        recognition_pipeline = create_pipeline("config.yaml")
        system_status["system_ready"] = True
        
        logger.info(f"✅ Training pipeline completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed for task {task_id}: {e}")
        background_tasks[task_id] = {"status": "failed", "error": str(e), "progress": 0}
    finally:
        system_status["training_in_progress"] = False

async def run_evaluation_pipeline(task_id: str):
    """Run the evaluation pipeline"""
    try:
        logger.info(f"🧪 Running evaluation pipeline for task {task_id}")
        
        background_tasks[task_id] = {"status": "running", "progress": 50}
        
        # Run evaluation
        success = ai_system.run_evaluation()
        
        if success:
            background_tasks[task_id] = {"status": "completed", "progress": 100}
            logger.info(f"✅ Evaluation completed for task {task_id}")
        else:
            background_tasks[task_id] = {"status": "failed", "error": "Evaluation failed", "progress": 0}
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed for task {task_id}: {e}")
        background_tasks[task_id] = {"status": "failed", "error": str(e), "progress": 0}
    finally:
        system_status["evaluation_in_progress"] = False

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )