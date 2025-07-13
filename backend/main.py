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

# Background task tracking and utility functions
background_tasks = {}

def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file for security and format compliance.
    
    Args:
        file: The uploaded file object
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Expected image, got: {file.content_type}"
        )
    
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
            )
    
    # Check filename
    if not file.filename or len(file.filename.strip()) == 0:
        raise HTTPException(status_code=400, detail="No filename provided")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem storage
    """
    import re
    # Remove any path separators and invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    name = name.strip(' .')
    # Ensure it's not empty
    if not name:
        name = f"image_{int(time.time())}"
    return name

# Pydantic models for API request/response validation
class SystemStatus(BaseModel):
    """System status information model.
    
    Attributes:
        initialized: Whether the AI system has been initialized
        training_in_progress: Whether model training is currently running
        evaluation_in_progress: Whether system evaluation is running
        last_error: Most recent error message, if any
        system_ready: Whether the system is ready for recognition tasks
    """
    initialized: bool
    training_in_progress: bool
    evaluation_in_progress: bool
    last_error: Optional[str]
    system_ready: bool

class ItemCreate(BaseModel):
    """Model for creating new items in the system.
    
    Attributes:
        item_id: Unique identifier for the item
        name: Human-readable name for the item
        description: Optional description of the item
        category: Optional category classification
    """
    item_id: str
    name: str
    description: Optional[str] = ""
    category: Optional[str] = ""

class RecognitionRequest(BaseModel):
    """Request model for image recognition.
    
    Attributes:
        image_data: Base64 encoded image data
        confidence_threshold: Minimum confidence for positive recognition (0.0-1.0)
    """
    image_data: str
    confidence_threshold: Optional[float] = 0.85

class TrainingConfig(BaseModel):
    """Configuration model for training parameters.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
    """
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 16
    learning_rate: Optional[float] = 0.0001

class ApiResponse(BaseModel):
    """Standardized API response model.
    
    Attributes:
        success: Whether the operation was successful
        message: Human-readable status message
        data: Optional response data
        error: Optional error details if success is False
    """
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the AI system on startup with comprehensive error handling.
    
    This function sets up the entire AI recognition system including:
    - Directory structure creation
    - AI system initialization
    - Recognition pipeline loading
    - System status updates
    """
    global ai_system, recognition_pipeline, system_status
    
    try:
        logger.info("🚀 Starting AI Recognition System Backend...")
        
        # Create necessary directories with proper error handling
        required_dirs = [
            "backend/logs",
            "backend/temp", 
            "backend/uploads",
            "data/raw",
            "data/augmented",
            "data/models",
            "checkpoints"
        ]
        
        for dir_path in required_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Created/verified directory: {dir_path}")
            except OSError as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                system_status["last_error"] = f"Directory creation failed: {str(e)}"
                return
        
        # Initialize AI system
        config_path = "config.yaml"
        if Path(config_path).exists():
            ai_system = AIRecognitionSystem(config_path)
            logger.info("✅ AI System initialized successfully")
            
            # Try to load recognition pipeline with detailed error handling
            try:
                recognition_pipeline = create_pipeline(config_path)
                system_status["system_ready"] = True
                logger.info("✅ Recognition pipeline loaded successfully")
                
                # Validate pipeline functionality
                if hasattr(recognition_pipeline, 'is_ready') and not recognition_pipeline.is_ready():
                    logger.warning("⚠️  Recognition pipeline loaded but not ready")
                    system_status["system_ready"] = False
                    
            except FileNotFoundError as e:
                logger.warning(f"⚠️  Model files not found: {e}")
                system_status["system_ready"] = False
                system_status["last_error"] = f"Model files missing: {str(e)}"
            except Exception as e:
                logger.warning(f"⚠️  Recognition pipeline not ready: {e}")
                system_status["system_ready"] = False
                system_status["last_error"] = f"Pipeline initialization failed: {str(e)}"
            
            system_status["initialized"] = True
            
        else:
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(f"❌ {error_msg}")
            system_status["last_error"] = error_msg
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        system_status["last_error"] = str(e)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check system health status.
    
    Returns comprehensive health information including system initialization
    status, backend readiness, and any recent errors.
    
    Returns:
        dict: Health status information with timestamp
    """
    try:
        is_healthy = system_status["initialized"] and system_status.get("last_error") is None
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "system_status": system_status,
            "api_version": "1.0.0",
            "uptime": datetime.now().isoformat()  # Could track actual uptime
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": f"Health check error: {str(e)}"
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
    """Get comprehensive list of all items in the system.
    
    Retrieves all items from the raw images directory with metadata
    including image counts and file information.
    
    Returns:
        ApiResponse: List of items with metadata
        
    Raises:
        HTTPException: If system is not initialized or directory access fails
    """
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
    """Create a new item in the system with validation.
    
    Creates a new item directory and metadata file. Validates that
    the item ID is unique and follows naming conventions.
    
    Args:
        item: ItemCreate model with item details
        
    Returns:
        ApiResponse: Success confirmation with item details
        
    Raises:
        HTTPException: If item already exists or creation fails
    """
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        raw_dir = Path(ai_system.config['data']['raw_images_dir'])
        item_dir = raw_dir / item.item_id
        
        # Validate item ID format
        if not item.item_id or not item.item_id.strip():
            raise HTTPException(status_code=400, detail="Item ID cannot be empty")
        
        if not item.item_id.replace('_', '').replace('-', '').isalnum():
            raise HTTPException(
                status_code=400, 
                detail="Item ID must contain only letters, numbers, hyphens, and underscores"
            )
        
        if item_dir.exists():
            raise HTTPException(status_code=409, detail=f"Item '{item.item_id}' already exists")
        
        # Create item directory with error handling
        try:
            item_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create item directory {item_dir}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create item directory: {str(e)}")
        
        # Create item metadata file
        metadata = {
            "item_id": item.item_id,
            "name": item.name,
            "description": item.description,
            "category": item.category,
            "created_at": datetime.now().isoformat()
        }
        
        # Create metadata file with error handling
        try:
            with open(item_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to create metadata file: {e}")
            # Clean up directory if metadata creation fails
            try:
                item_dir.rmdir()
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to create item metadata: {str(e)}")
        
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
    """Upload images for an item with comprehensive validation.
    
    Accepts multiple image files and stores them in the item's directory.
    Validates file types, sizes, and names for security.
    
    Args:
        item_id: The unique identifier of the item
        files: List of image files to upload
        
    Returns:
        ApiResponse: Upload results with file information
        
    Raises:
        HTTPException: If item not found, validation fails, or upload errors
    """
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        raw_dir = Path(ai_system.config['data']['raw_images_dir'])
        item_dir = raw_dir / item_id
        
        if not item_dir.exists():
            raise HTTPException(status_code=404, detail="Item not found")
        
        uploaded_files = []
        failed_files = []
        
        for file in files:
            try:
                # Validate each file
                validate_file_upload(file)
                
                # Generate unique, sanitized filename
                timestamp = int(time.time() * 1000)
                safe_filename = sanitize_filename(file.filename)
                filename = f"{timestamp}_{safe_filename}"
                file_path = item_dir / filename
                
                # Save file with error handling
                try:
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    # Verify file was saved correctly
                    if not file_path.exists() or file_path.stat().st_size == 0:
                        raise OSError("File was not saved correctly")
                    
                    uploaded_files.append(filename)
                    logger.debug(f"Successfully uploaded: {filename}")
                    
                except OSError as e:
                    logger.error(f"Failed to save file {filename}: {e}")
                    failed_files.append({"filename": file.filename, "error": str(e)})
                    # Clean up partial file
                    if file_path.exists():
                        try:
                            file_path.unlink()
                        except OSError:
                            pass
                            
            except HTTPException as e:
                # File validation failed
                failed_files.append({"filename": file.filename, "error": e.detail})
                logger.warning(f"File validation failed for {file.filename}: {e.detail}")
            except Exception as e:
                # Unexpected error
                failed_files.append({"filename": file.filename, "error": f"Unexpected error: {str(e)}"})
                logger.error(f"Unexpected error processing {file.filename}: {e}")
        
        # Prepare response with detailed results
        total_files = len(uploaded_files) + len(failed_files)
        success_count = len(uploaded_files)
        
        if success_count == 0 and failed_files:
            # All files failed
            raise HTTPException(
                status_code=400,
                detail=f"Failed to upload any files. Errors: {failed_files}"
            )
        
        response_data = {
            "uploaded_files": uploaded_files,
            "total_files": total_files,
            "successful_uploads": success_count,
            "failed_uploads": len(failed_files)
        }
        
        if failed_files:
            response_data["failed_files"] = failed_files
        
        message = f"Uploaded {success_count}/{total_files} images successfully"
        if failed_files:
            message += f" ({len(failed_files)} files failed)"
        
        logger.info(f"✅ {message} for item {item_id}")
        return ApiResponse(
            success=True,
            message=message,
            data=response_data
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
    """Recognize an image and return the prediction result.
    
    This endpoint accepts a base64-encoded image and performs recognition
    using the trained model pipeline. Returns detailed recognition results
    including confidence scores and processing metrics.
    
    Args:
        request: RecognitionRequest containing image data and parameters
        
    Returns:
        ApiResponse: Recognition results with item ID, confidence, and metrics
        
    Raises:
        HTTPException: If recognition pipeline is not ready or request is invalid
    """
    try:
        if not recognition_pipeline:
            raise HTTPException(status_code=503, detail="Recognition pipeline not ready")
        
        # Validate and decode base64 image
        try:
            if not request.image_data:
                raise HTTPException(status_code=400, detail="No image data provided")
            
            # Remove data URL prefix if present
            image_data_clean = request.image_data
            if 'data:image' in image_data_clean:
                image_data_clean = image_data_clean.split(',')[1]
            
            image_data = base64.b64decode(image_data_clean)
            
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Empty image data")
                
        except base64.binascii.Error as e:
            logger.error(f"Base64 decode error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        except Exception as e:
            logger.error(f"Image data processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process image data: {str(e)}")
        
        # Save to temporary file with proper error handling
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
                
            # Validate the saved file
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise HTTPException(status_code=400, detail="Failed to save image data")
                
        except OSError as e:
            logger.error(f"File system error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")
        
        try:
            # Run recognition with timeout and error handling
            start_time = time.time()
            
            try:
                result = recognition_pipeline.recognize(temp_path)
                processing_time = time.time() - start_time
                
                if not result:
                    raise HTTPException(status_code=500, detail="Recognition pipeline returned no result")
                    
            except Exception as recognition_error:
                processing_time = time.time() - start_time
                logger.error(f"Recognition pipeline error: {recognition_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Recognition failed: {str(recognition_error)}"
                )
            
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
            # Clean up temp file with error handling
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
                except OSError as e:
                    logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
        
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