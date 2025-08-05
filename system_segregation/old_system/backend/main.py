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
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Add project root to Python path for imports
sys.path.insert(0, str(project_root))

# Import our AI system components
try:
    from main import AIRecognitionSystem
    from src.inference.recognize import create_pipeline, RecognitionResult
except ImportError as e:
    logger.error(f"Failed to import AI system components: {e}")
    # Create fallback implementations
    AIRecognitionSystem = None
    create_pipeline = None
    RecognitionResult = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
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

# Training control
training_stop_flag = False

# Background task tracking and utility functions
background_tasks = {}

# Auto-training configuration
auto_training_config = {
    "enabled": True,
    "min_images_threshold": 5,  # Minimum images per item before auto-training
    "batch_delay_minutes": 30,  # Wait 30 minutes before training to batch uploads
    "last_training_trigger": None,
    "pending_items": set(),  # Items that need retraining
    "state_file": "backend/auto_training_state.json",  # Persistent state file
    "training_in_progress_file": "backend/training_in_progress.lock"  # Training lock file
}

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

def check_auto_training_trigger(item_id: str) -> None:
    """Check if auto-training should be triggered for an item"""
    if not auto_training_config["enabled"]:
        return
    
    try:
        # Count images for this item
        raw_dir = Path(ai_system.config['data']['raw_images_dir'])
        item_dir = raw_dir / item_id
        
        if not item_dir.exists():
            return
            
        # Count image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_count = 0
        for ext in image_extensions:
            image_count += len(list(item_dir.glob(f"*{ext}")))
            image_count += len(list(item_dir.glob(f"*{ext.upper()}")))
        
        logger.info(f"📊 Item {item_id} now has {image_count} images")
        
        # Check if item meets minimum threshold
        if image_count >= auto_training_config["min_images_threshold"]:
            auto_training_config["pending_items"].add(item_id)
            logger.info(f"🎯 Item {item_id} marked for auto-training (has {image_count} images)")
            
            # Schedule auto-training
            schedule_auto_training()
        
    except Exception as e:
        logger.error(f"Auto-training check failed for {item_id}: {e}")

def schedule_auto_training() -> None:
    """Schedule auto-training with batch delay"""
    if not auto_training_config["pending_items"]:
        return
    
    import threading
    import datetime
    
    # Cancel existing timer if any
    if hasattr(schedule_auto_training, '_timer'):
        schedule_auto_training._timer.cancel()
    
    delay_seconds = auto_training_config["batch_delay_minutes"] * 60
    
    def trigger_training():
        if auto_training_config["pending_items"]:
            pending_count = len(auto_training_config["pending_items"])
            logger.info(f"🚀 Auto-triggering training for {pending_count} items: {list(auto_training_config['pending_items'])}")
            
            # Trigger training in background
            task_id = f"auto_training_{int(time.time())}"
            asyncio.create_task(run_auto_training_pipeline(task_id))
            
            # Clear pending items
            auto_training_config["pending_items"].clear()
            auto_training_config["last_training_trigger"] = datetime.datetime.now().isoformat()
    
    # Schedule training with delay
    schedule_auto_training._timer = threading.Timer(delay_seconds, trigger_training)
    schedule_auto_training._timer.start()
    
    logger.info(f"⏰ Auto-training scheduled for {len(auto_training_config['pending_items'])} items in {auto_training_config['batch_delay_minutes']} minutes")

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
                if create_pipeline:
                    # Load with our optimized recognition system
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Create config for our state-of-the-art recognition system
                    pipeline_config = {
                        'model_path': 'checkpoints/best_model_DISABLED.pth',
                        'index_path': 'data/models/faiss_index.bin',
                        'metadata_path': 'data/models/index_metadata.pkl',
                        'threshold': 0.85,
                        'clip_model': config.get('model', {}).get('clip_variant', 'ViT-L/14'),
                        'embedding_dim': config.get('model', {}).get('embedding_dim', 512),
                        'cache_size': 1000,
                        'confidence_threshold': 0.85,
                        'high_confidence_threshold': 0.95,
                        'batch_confidence_threshold': 0.9,
                        'mobile_mode': False
                    }
                    
                    # Import and create our optimized pipeline
                    from src.inference.recognize import RecognitionPipeline
                    recognition_pipeline = RecognitionPipeline(pipeline_config)
                    
                    system_status["system_ready"] = True
                    logger.info("✅ State-of-the-art recognition pipeline loaded successfully")
                else:
                    logger.warning("⚠️  Recognition pipeline not available (import failed)")
                    system_status["system_ready"] = False
                
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
            
            # Check for interrupted training
            interrupted_state = check_interrupted_training()
            if interrupted_state:
                logger.warning(f"🚨 Training was interrupted. Manual intervention required.")
                system_status["training_interrupted"] = True
                system_status["training_required"] = True
            
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
        
        if AIRecognitionSystem:
            ai_system = AIRecognitionSystem("config.yaml")
        
        if create_pipeline:
            # Load with our optimized recognition system
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            pipeline_config = {
                'model_path': 'checkpoints/best_model_DISABLED.pth',
                'index_path': 'data/models/faiss_index.bin',
                'metadata_path': 'data/models/index_metadata.pkl',
                'threshold': 0.85,
                'clip_model': config.get('model', {}).get('clip_variant', 'ViT-L/14'),
                'embedding_dim': config.get('model', {}).get('embedding_dim', 512),
                'cache_size': 1000,
                'confidence_threshold': 0.85,
                'high_confidence_threshold': 0.95,
                'batch_confidence_threshold': 0.9,
                'mobile_mode': False
            }
            
            from src.inference.recognize import RecognitionPipeline
            recognition_pipeline = RecognitionPipeline(pipeline_config)
        
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
        
        # Trigger auto-training check if images were successfully uploaded
        if success_count > 0:
            check_auto_training_trigger(item_id)
        
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
async def start_training(bg_tasks: BackgroundTasks, config: TrainingConfig = None):
    """Start model training in the background"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        if system_status["training_in_progress"]:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        # Start training in background
        task_id = f"training_{int(time.time())}"
        bg_tasks.add_task(run_training_pipeline, task_id, config)
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

@app.post("/api/train/stop")
async def stop_training():
    """Stop the current training process gracefully"""
    global training_stop_flag
    
    try:
        if not system_status["training_in_progress"]:
            return ApiResponse(
                success=False,
                message="No training in progress",
                data={"training_in_progress": False}
            )
        
        # Set stop flag for graceful shutdown
        training_stop_flag = True
        logger.info("🛑 Training stop requested - setting stop flag")
        
        return ApiResponse(
            success=True,
            message="Training stop requested - will stop gracefully at next checkpoint",
            data={"stop_requested": True}
        )
        
    except Exception as e:
        error_msg = f"Failed to stop training: {str(e)}"
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
            "stop_requested": training_stop_flag,
            "tasks": background_tasks
        }
    )

# Evaluation endpoints
@app.post("/api/evaluate")
async def start_evaluation(bg_tasks: BackgroundTasks):
    """Start system evaluation in the background"""
    try:
        if not ai_system:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        if system_status["evaluation_in_progress"]:
            raise HTTPException(status_code=400, detail="Evaluation already in progress")
        
        task_id = f"evaluation_{int(time.time())}"
        bg_tasks.add_task(run_evaluation_pipeline, task_id)
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
# Auto-training control endpoints
@app.get("/api/auto-training/status")
async def get_auto_training_status():
    """Get auto-training configuration and status"""
    return ApiResponse(
        success=True,
        message="Auto-training status retrieved",
        data={
            "config": auto_training_config,
            "pending_items": list(auto_training_config["pending_items"]),
            "active_training": system_status["training_in_progress"]
        }
    )

@app.post("/api/auto-training/config")
async def update_auto_training_config(
    enabled: bool = None,
    min_images_threshold: int = None,
    batch_delay_minutes: int = None
):
    """Update auto-training configuration"""
    if enabled is not None:
        auto_training_config["enabled"] = enabled
    if min_images_threshold is not None:
        auto_training_config["min_images_threshold"] = min_images_threshold
    if batch_delay_minutes is not None:
        auto_training_config["batch_delay_minutes"] = batch_delay_minutes
    
    logger.info(f"🔧 Auto-training config updated: {auto_training_config}")
    
    return ApiResponse(
        success=True,
        message="Auto-training configuration updated",
        data=auto_training_config
    )

@app.post("/api/auto-training/trigger")
async def force_auto_training():
    """Manually trigger auto-training for all pending items"""
    if not auto_training_config["pending_items"]:
        return ApiResponse(
            success=False,
            message="No items pending for training",
            data={"pending_count": 0}
        )
    
    # Cancel scheduled training
    if hasattr(schedule_auto_training, '_timer'):
        schedule_auto_training._timer.cancel()
    
    # Trigger immediately
    task_id = f"manual_auto_training_{int(time.time())}"
    asyncio.create_task(run_auto_training_pipeline(task_id))
    
    pending_count = len(auto_training_config["pending_items"])
    auto_training_config["pending_items"].clear()
    
    return ApiResponse(
        success=True,
        message=f"Auto-training triggered for {pending_count} items",
        data={"task_id": task_id, "items_count": pending_count}
    )

@app.post("/api/training/manual")
async def manual_training(manual_config: dict):
    """Manually trigger training for specific items or all items"""
    try:
        if system_status["training_in_progress"]:
            return ApiResponse(
                success=False,
                message="Training already in progress. Please wait for completion.",
                data={"training_in_progress": True}
            )
        
        # Save training state for persistence
        training_state_file = Path("backend/training_state.json")
        training_state = {
            "in_progress": True,
            "started_at": datetime.now().isoformat(),
            "config": manual_config,
            "type": "manual",
            "task_id": None
        }
        
        if manual_config.get("batch", False):
            # Train all items
            task_id = f"manual_batch_training_{int(time.time())}"
            training_state["task_id"] = task_id
            
            # Save state to disk
            with open(training_state_file, 'w') as f:
                json.dump(training_state, f, indent=2)
            
            asyncio.create_task(run_manual_training_pipeline(task_id, manual_config))
            
            # Count available items
            items_dir = Path("data/raw")
            items_count = len([d for d in items_dir.iterdir() if d.is_dir() and any(d.glob("*.jpg"))])
            
            return ApiResponse(
                success=True,
                message=f"Manual batch training started for all items",
                data={"task_id": task_id, "items_count": items_count, "type": "batch"}
            )
        else:
            # Train specific item
            item_id = manual_config.get("item_id")
            if not item_id:
                return ApiResponse(
                    success=False,
                    message="item_id is required for single item training"
                )
            
            task_id = f"manual_item_training_{item_id}_{int(time.time())}"
            training_state["task_id"] = task_id
            training_state["item_id"] = item_id
            
            # Save state to disk
            with open(training_state_file, 'w') as f:
                json.dump(training_state, f, indent=2)
            
            asyncio.create_task(run_manual_training_pipeline(task_id, manual_config))
            
            return ApiResponse(
                success=True,
                message=f"Manual training started for item {item_id}",
                data={"task_id": task_id, "item_id": item_id, "type": "single"}
            )
        
    except Exception as e:
        logger.error(f"Failed to start manual training: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to start manual training: {str(e)}"
        )

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
    global training_stop_flag
    
    try:
        logger.info(f"🏃 Running training pipeline for task {task_id}")
        training_stop_flag = False  # Reset stop flag
        
        # Update task status
        background_tasks[task_id] = {"status": "preparing", "progress": 10}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run data preparation
        if not ai_system.run_data_preparation():
            raise Exception("Data preparation failed")
        background_tasks[task_id] = {"status": "data_prepared", "progress": 30}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run feature extraction
        if not ai_system.run_feature_extraction():
            raise Exception("Feature extraction failed")
        background_tasks[task_id] = {"status": "features_extracted", "progress": 50}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run training
        if not ai_system.run_training():
            raise Exception("Model training failed")
        background_tasks[task_id] = {"status": "training_complete", "progress": 80}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Build recognition index
        if not ai_system.build_recognition_index():
            raise Exception("Index building failed")
        background_tasks[task_id] = {"status": "completed", "progress": 100}
        
        # Reload recognition pipeline with optimized system
        global recognition_pipeline
        if create_pipeline:
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            pipeline_config = {
                'model_path': 'checkpoints/best_model_DISABLED.pth',
                'index_path': 'data/models/faiss_index.bin',
                'metadata_path': 'data/models/index_metadata.pkl',
                'threshold': 0.85,
                'clip_model': config.get('model', {}).get('clip_variant', 'ViT-L/14'),
                'embedding_dim': config.get('model', {}).get('embedding_dim', 512),
                'cache_size': 1000,
                'confidence_threshold': 0.85,
                'high_confidence_threshold': 0.95,
                'batch_confidence_threshold': 0.9,
                'mobile_mode': False
            }
            
            from src.inference.recognize import RecognitionPipeline
            recognition_pipeline = RecognitionPipeline(pipeline_config)
            system_status["system_ready"] = True
        
        logger.info(f"✅ Training pipeline completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed for task {task_id}: {e}")
        background_tasks[task_id] = {"status": "failed", "error": str(e), "progress": 0}
    finally:
        system_status["training_in_progress"] = False
        training_stop_flag = False  # Reset stop flag

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

async def run_auto_training_pipeline(task_id: str):
    """Run automatic training pipeline triggered by image uploads"""
    global training_stop_flag
    
    try:
        logger.info(f"🤖 Running auto-training pipeline for task {task_id}")
        
        # Check if training is already in progress
        if system_status["training_in_progress"]:
            logger.info("⏸️  Training already in progress, skipping auto-training")
            background_tasks[task_id] = {"status": "skipped", "reason": "training_in_progress", "progress": 0}
            return
        
        # Update task status
        background_tasks[task_id] = {"status": "preparing", "progress": 10}
        system_status["training_in_progress"] = True
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run data preparation (includes augmentation)
        if not ai_system.run_data_preparation():
            raise Exception("Data preparation failed")
        background_tasks[task_id] = {"status": "data_prepared", "progress": 30}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run feature extraction
        if not ai_system.run_feature_extraction():
            raise Exception("Feature extraction failed")
        background_tasks[task_id] = {"status": "features_extracted", "progress": 50}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run training
        if not ai_system.run_training():
            raise Exception("Model training failed")
        background_tasks[task_id] = {"status": "training_complete", "progress": 80}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Build recognition index
        if not ai_system.build_recognition_index():
            raise Exception("Index building failed")
        background_tasks[task_id] = {"status": "completed", "progress": 100}
        
        # Reload recognition pipeline
        global recognition_pipeline
        if create_pipeline:
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            pipeline_config = {
                'model_path': 'checkpoints/best_model_DISABLED.pth',
                'index_path': 'data/models/faiss_index.bin',
                'metadata_path': 'data/models/index_metadata.pkl',
                'threshold': 0.85,
                'clip_model': config.get('model', {}).get('clip_variant', 'ViT-L/14'),
                'embedding_dim': config.get('model', {}).get('embedding_dim', 512),
                'cache_size': 1000,
                'confidence_threshold': 0.85,
                'high_confidence_threshold': 0.95,
                'batch_confidence_threshold': 0.9,
                'mobile_mode': False
            }
            
            from src.inference.recognize import RecognitionPipeline
            recognition_pipeline = RecognitionPipeline(pipeline_config)
            system_status["system_ready"] = True
        
        logger.info(f"✅ Auto-training pipeline completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Auto-training pipeline failed for task {task_id}: {e}")
        background_tasks[task_id] = {"status": "failed", "error": str(e), "progress": 0}
    finally:
        system_status["training_in_progress"] = False
        training_stop_flag = False  # Reset stop flag

async def run_manual_training_pipeline(task_id: str, manual_config: dict):
    """Run manual training pipeline for specific items"""
    global training_stop_flag
    training_state_file = Path("backend/training_state.json")
    
    try:
        logger.info(f"🎯 Running manual training pipeline for task {task_id}")
        
        # Update task status
        background_tasks[task_id] = {"status": "preparing", "progress": 10}
        system_status["training_in_progress"] = True
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run data preparation
        if not ai_system.run_data_preparation():
            raise Exception("Data preparation failed")
        background_tasks[task_id] = {"status": "data_prepared", "progress": 30}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run feature extraction
        if not ai_system.run_feature_extraction():
            raise Exception("Feature extraction failed")
        background_tasks[task_id] = {"status": "features_extracted", "progress": 50}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Run training
        if not ai_system.run_training():
            raise Exception("Model training failed")
        background_tasks[task_id] = {"status": "training_complete", "progress": 80}
        
        # Check for stop request
        if training_stop_flag:
            raise Exception("Training stopped by user request")
        
        # Build recognition index
        if not ai_system.build_recognition_index():
            raise Exception("Index building failed")
        background_tasks[task_id] = {"status": "completed", "progress": 100}
        
        # Reload recognition pipeline
        global recognition_pipeline
        if create_pipeline:
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            pipeline_config = {
                'model_path': 'checkpoints/best_model_DISABLED.pth',
                'index_path': 'data/models/faiss_index.bin',
                'metadata_path': 'data/models/index_metadata.pkl',
                'threshold': 0.85,
                'clip_model': config.get('model', {}).get('clip_variant', 'ViT-L/14'),
                'embedding_dim': config.get('model', {}).get('embedding_dim', 512),
                'cache_size': 1000,
                'confidence_threshold': 0.85,
                'high_confidence_threshold': 0.95,
                'batch_confidence_threshold': 0.9,
                'mobile_mode': False
            }
            
            from src.inference.recognize import RecognitionPipeline
            recognition_pipeline = RecognitionPipeline(pipeline_config)
            system_status["system_ready"] = True
        
        # Clear training state on successful completion
        if training_state_file.exists():
            training_state_file.unlink()
        
        logger.info(f"✅ Manual training pipeline completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Manual training pipeline failed for task {task_id}: {e}")
        background_tasks[task_id] = {"status": "failed", "error": str(e), "progress": 0}
        
        # Mark training as requiring restart
        if training_state_file.exists():
            try:
                with open(training_state_file, 'r') as f:
                    state = json.load(f)
                state["in_progress"] = False
                state["failed"] = True
                state["failed_at"] = datetime.now().isoformat()
                state["error"] = str(e)
                with open(training_state_file, 'w') as f:
                    json.dump(state, f, indent=2)
            except Exception as state_error:
                logger.error(f"Failed to update training state: {state_error}")
    finally:
        system_status["training_in_progress"] = False
        training_stop_flag = False  # Reset stop flag

def check_interrupted_training():
    """Check for interrupted training on startup"""
    training_state_file = Path("backend/training_state.json")
    
    if not training_state_file.exists():
        return None
    
    try:
        with open(training_state_file, 'r') as f:
            state = json.load(f)
        
        # If training was in progress but we're starting fresh, it was interrupted
        if state.get("in_progress", False) and not state.get("failed", False):
            logger.warning(f"🚨 Detected interrupted training from {state.get('started_at')}")
            
            # Mark as needing restart
            state["in_progress"] = False
            state["interrupted"] = True
            state["interrupted_at"] = datetime.now().isoformat()
            
            with open(training_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            return state
        
        # If training failed or was completed, remove the state file
        elif not state.get("in_progress", False):
            training_state_file.unlink()
            return None
    
    except Exception as e:
        logger.error(f"Failed to check interrupted training state: {e}")
        # Remove corrupted state file
        try:
            training_state_file.unlink()
        except:
            pass
        return None

@app.get("/api/training/status")
async def get_training_status():
    """Get training status including interrupted training detection"""
    try:
        training_state_file = Path("backend/training_state.json")
        
        status = {
            "training_in_progress": system_status.get("training_in_progress", False),
            "interrupted_training": False,
            "training_required": False,
            "background_tasks": background_tasks
        }
        
        if training_state_file.exists():
            with open(training_state_file, 'r') as f:
                state = json.load(f)
            
            if state.get("interrupted", False) or state.get("failed", False):
                status["interrupted_training"] = True
                status["training_required"] = True
                status["last_attempt"] = state
        
        return ApiResponse(
            success=True,
            message="Training status retrieved",
            data=status
        )
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to get training status: {str(e)}"
        )

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )