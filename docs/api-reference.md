# API Reference - AI Recognition System

## Overview

The AI Recognition System provides a RESTful API built with FastAPI. This document covers all available endpoints, request/response formats, and integration examples.

## Base URL

```
Development: http://localhost:8000
Production: http://your-domain.com:8000
```

## API Endpoints

### 1. Health Check

#### GET /health
Check if the API server is running and healthy.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

### 2. System Status

#### GET /status
Get detailed system status including model loading state, index information, and performance metrics.

**Request:**
```bash
curl -X GET "http://localhost:8000/status"
```

**Response:**
```json
{
  "system_status": "ready",
  "models_loaded": true,
  "index_info": {
    "total_vectors": 11,
    "dimensions": 1536,
    "index_type": "Flat"
  },
  "performance": {
    "avg_recognition_time_ms": 300,
    "total_recognitions": 150,
    "accuracy_rate": 1.0
  },
  "memory_usage_mb": 1890
}
```

### 3. Image Recognition

#### POST /recognize
Recognize an item from an uploaded image.

**Request:**
```bash
curl -X POST "http://localhost:8000/recognize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"
```

**Request Parameters:**
- `file` (required): Image file (JPG, PNG, JPEG formats)
- `threshold` (optional): Override confidence threshold (0.0-1.0)

**Response (Successful Recognition):**
```json
{
  "status": "success",
  "item_id": "item_001",
  "confidence": 1.414,
  "processing_time_ms": 285,
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "stage1_candidates": 4,
    "stage2_refinement": true,
    "stage3_verification": false
  }
}
```

**Response (Unknown Item):**
```json
{
  "status": "success",
  "item_id": "unknown",
  "confidence": 0.45,
  "processing_time_ms": 92,
  "timestamp": "2024-01-15T10:30:00Z",
  "reason": "Confidence below threshold"
}
```

**Response (Error):**
```json
{
  "status": "error",
  "error_code": "INVALID_IMAGE",
  "message": "Unable to process image file",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 4. Batch Recognition

#### POST /recognize/batch
Recognize multiple images in a single request.

**Request:**
```bash
curl -X POST "http://localhost:8000/recognize/batch" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "filename": "image1.jpg",
      "item_id": "item_001",
      "confidence": 1.414,
      "processing_time_ms": 285
    },
    {
      "filename": "image2.jpg", 
      "item_id": "unknown",
      "confidence": 0.45,
      "processing_time_ms": 92
    },
    {
      "filename": "image3.jpg",
      "item_id": "item_003",
      "confidence": 1.386,
      "processing_time_ms": 310
    }
  ],
  "total_processing_time_ms": 687,
  "processed_count": 3
}
```

### 5. Item Management

#### GET /items
List all items in the recognition system.

**Request:**
```bash
curl -X GET "http://localhost:8000/items"
```

**Response:**
```json
{
  "status": "success",
  "items": [
    {
      "item_id": "item_001",
      "name": "Red Widget",
      "category": "Widgets",
      "description": "Standard red widget",
      "image_count": 3,
      "created_at": "2024-01-10T09:00:00Z"
    },
    {
      "item_id": "item_002",
      "name": "Blue Component",
      "category": "Components", 
      "description": "Blue electronic component",
      "image_count": 3,
      "created_at": "2024-01-11T10:15:00Z"
    }
  ],
  "total_items": 4
}
```

#### GET /items/{item_id}
Get detailed information about a specific item.

**Request:**
```bash
curl -X GET "http://localhost:8000/items/item_001"
```

**Response:**
```json
{
  "status": "success",
  "item": {
    "item_id": "item_001",
    "name": "Red Widget",
    "category": "Widgets",
    "description": "Standard red widget",
    "images": [
      {
        "filename": "Copy of IMG_8388.JPG",
        "size_bytes": 2457600,
        "resolution": "4032x3024",
        "added_at": "2024-01-10T09:00:00Z"
      }
    ],
    "recognition_stats": {
      "total_recognitions": 25,
      "avg_confidence": 1.41,
      "last_recognized": "2024-01-15T10:25:00Z"
    }
  }
}
```

#### POST /items
Add a new item to the system.

**Request:**
```bash
curl -X POST "http://localhost:8000/items" \
     -H "Content-Type: multipart/form-data" \
     -F "item_id=item_005" \
     -F "name=Green Gadget" \
     -F "category=Gadgets" \
     -F "description=New green gadget" \
     -F "images=@image1.jpg" \
     -F "images=@image2.jpg"
```

**Response:**
```json
{
  "status": "success",
  "message": "Item added successfully",
  "item_id": "item_005",
  "images_processed": 2,
  "requires_retraining": true
}
```

#### DELETE /items/{item_id}
Remove an item from the system.

**Request:**
```bash
curl -X DELETE "http://localhost:8000/items/item_005"
```

**Response:**
```json
{
  "status": "success",
  "message": "Item deleted successfully",
  "item_id": "item_005",
  "requires_retraining": true
}
```

### 6. Training Management

#### POST /training/start
Start model training with current data.

**Request:**
```bash
curl -X POST "http://localhost:8000/training/start" \
     -H "Content-Type: application/json" \
     -d '{
       "epochs": 40,
       "batch_size": 24,
       "learning_rate": 0.0003
     }'
```

**Response:**
```json
{
  "status": "success", 
  "message": "Training started",
  "training_id": "train_20240115_103000",
  "estimated_duration_minutes": 120,
  "configuration": {
    "epochs": 40,
    "batch_size": 24,
    "learning_rate": 0.0003,
    "total_samples": 1600
  }
}
```

#### GET /training/status
Get current training status.

**Request:**
```bash
curl -X GET "http://localhost:8000/training/status"
```

**Response (Training in Progress):**
```json
{
  "status": "training",
  "training_id": "train_20240115_103000", 
  "progress": {
    "current_epoch": 15,
    "total_epochs": 40,
    "progress_percent": 37.5,
    "estimated_remaining_minutes": 78
  },
  "metrics": {
    "current_loss": 0.245,
    "current_accuracy": 0.932,
    "best_accuracy": 0.945
  }
}
```

**Response (Training Complete):**
```json
{
  "status": "completed",
  "training_id": "train_20240115_103000",
  "completion_time": "2024-01-15T12:30:00Z",
  "final_metrics": {
    "final_loss": 0.156,
    "final_accuracy": 0.967,
    "training_duration_minutes": 118
  },
  "model_saved": true
}
```

#### POST /training/stop
Stop current training process.

**Request:**
```bash
curl -X POST "http://localhost:8000/training/stop"
```

**Response:**
```json
{
  "status": "success",
  "message": "Training stopped",
  "training_id": "train_20240115_103000",
  "stopped_at_epoch": 23
}
```

### 7. Evaluation

#### POST /evaluation/run
Run system evaluation on test data.

**Request:**
```bash
curl -X POST "http://localhost:8000/evaluation/run"
```

**Response:**
```json
{
  "status": "success",
  "evaluation_id": "eval_20240115_140000",
  "results": {
    "overall_accuracy": 1.0,
    "unknown_detection_rate": 1.0,
    "avg_processing_time_ms": 300,
    "per_item_accuracy": {
      "item_001": 1.0,
      "item_002": 1.0,
      "item_003": 1.0,
      "item_004": 1.0
    },
    "confidence_distribution": {
      "mean": 1.395,
      "std": 0.025,
      "min": 1.365,
      "max": 1.444
    }
  },
  "test_summary": {
    "total_tests": 5,
    "known_items_tested": 4,
    "unknown_items_tested": 1,
    "all_tests_passed": true
  }
}
```

### 8. Configuration

#### GET /config
Get current system configuration.

**Request:**
```bash
curl -X GET "http://localhost:8000/config"
```

**Response:**
```json
{
  "status": "success",
  "configuration": {
    "confidence_threshold": 0.90,
    "min_stage1_confidence": 0.7,
    "features": {
      "clip_model": "ViT-L/14",
      "total_dimensions": 1536,
      "batch_size": 32
    },
    "training": {
      "epochs": 40,
      "batch_size": 24,
      "learning_rate": 0.0003
    }
  }
}
```

#### PUT /config
Update system configuration.

**Request:**
```bash
curl -X PUT "http://localhost:8000/config" \
     -H "Content-Type: application/json" \
     -d '{
       "confidence_threshold": 0.95,
       "min_stage1_confidence": 0.8
     }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration updated",
  "updated_fields": ["confidence_threshold", "min_stage1_confidence"],
  "requires_restart": false
}
```

## WebSocket Endpoints

### Real-time Status Updates

#### WS /ws/status
WebSocket connection for real-time system status updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/status');

ws.onmessage = function(event) {
    const status = JSON.parse(event.data);
    console.log('System status:', status);
};
```

**Message Format:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "system_status": "ready",
  "memory_usage_mb": 1890,
  "active_recognitions": 0,
  "total_recognitions": 150,
  "avg_response_time_ms": 285
}
```

### Training Progress

#### WS /ws/training
WebSocket connection for real-time training progress updates.

**Message Format:**
```json
{
  "training_id": "train_20240115_103000",
  "epoch": 15,
  "loss": 0.245,
  "accuracy": 0.932,
  "progress_percent": 37.5,
  "estimated_remaining_minutes": 78
}
```

## Error Codes

### Standard HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (item/endpoint not found)
- `413`: Payload Too Large (file too big)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error
- `503`: Service Unavailable (system not ready)

### Custom Error Codes
```json
{
  "error_codes": {
    "INVALID_IMAGE": "Image file cannot be processed",
    "MODEL_NOT_LOADED": "Recognition models not ready",
    "TRAINING_IN_PROGRESS": "Cannot perform operation while training",
    "INSUFFICIENT_DATA": "Not enough training data",
    "ITEM_NOT_FOUND": "Specified item does not exist",
    "DUPLICATE_ITEM": "Item ID already exists",
    "INDEX_NOT_READY": "Recognition index not built",
    "CONFIG_VALIDATION_ERROR": "Configuration validation failed"
  }
}
```

## Authentication

### API Key Authentication (Optional)
```bash
# If API key authentication is enabled
curl -X POST "http://localhost:8000/recognize" \
     -H "X-API-Key: your-api-key-here" \
     -F "file=@image.jpg"
```

### JWT Authentication (Optional)
```bash
# If JWT authentication is enabled
curl -X POST "http://localhost:8000/recognize" \
     -H "Authorization: Bearer your-jwt-token" \
     -F "file=@image.jpg"
```

## Rate Limiting

### Default Limits
- Recognition requests: 100 per minute
- Training requests: 1 per hour
- Configuration updates: 10 per minute

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1642248600
```

## Integration Examples

### Python Client
```python
import requests
import json

class AIRecognitionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def recognize_image(self, image_path, threshold=None):
        url = f"{self.base_url}/recognize"
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'threshold': threshold} if threshold else {}
            
            response = requests.post(url, files=files, params=params)
            return response.json()
    
    def get_system_status(self):
        url = f"{self.base_url}/status"
        response = requests.get(url)
        return response.json()
    
    def add_item(self, item_id, name, images, category=None, description=None):
        url = f"{self.base_url}/items"
        
        data = {
            'item_id': item_id,
            'name': name,
            'category': category or '',
            'description': description or ''
        }
        
        files = [('images', open(img, 'rb')) for img in images]
        
        try:
            response = requests.post(url, data=data, files=files)
            return response.json()
        finally:
            for _, f in files:
                f.close()

# Usage example
client = AIRecognitionClient()

# Recognize an image
result = client.recognize_image('test_image.jpg')
print(f"Recognized: {result['item_id']} (confidence: {result['confidence']})")

# Check system status
status = client.get_system_status()
print(f"System ready: {status['system_status'] == 'ready'}")
```

### JavaScript Client
```javascript
class AIRecognitionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async recognizeImage(imageFile, threshold = null) {
        const formData = new FormData();
        formData.append('file', imageFile);
        
        const params = threshold ? `?threshold=${threshold}` : '';
        const url = `${this.baseUrl}/recognize${params}`;
        
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    async getSystemStatus() {
        const response = await fetch(`${this.baseUrl}/status`);
        return await response.json();
    }
    
    // WebSocket connection for real-time updates
    connectToStatusUpdates(callback) {
        const ws = new WebSocket(`ws://localhost:8000/ws/status`);
        
        ws.onmessage = (event) => {
            const status = JSON.parse(event.data);
            callback(status);
        };
        
        return ws;
    }
}

// Usage example
const client = new AIRecognitionClient();

// File input handler
document.getElementById('imageInput').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const result = await client.recognizeImage(file);
        console.log('Recognition result:', result);
    }
});

// Real-time status updates
client.connectToStatusUpdates((status) => {
    console.log('System status update:', status);
});
```

## Performance Considerations

### Request Optimization
- Use batch recognition for multiple images
- Compress images before upload (while maintaining quality)
- Use WebSocket connections for real-time updates
- Cache configuration data on client side

### Response Times
- Single image recognition: ~300ms
- Batch recognition: ~200ms per image (parallelized)
- System status: <10ms
- Configuration updates: <50ms

### File Size Limits
- Maximum image size: 10MB per file
- Supported formats: JPG, JPEG, PNG
- Recommended resolution: 1080p to 4K
- Batch upload limit: 50 files per request