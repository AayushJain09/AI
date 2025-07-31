# 🔄 Unified Preprocessing System Flow Diagrams

## Current System vs Unified System

### **CURRENT SYSTEM** (Scattered Architecture)
```mermaid
graph TD
    subgraph "Training Flow"
        A1[Raw Images] --> B1[prepare.py]
        B1 --> C1[Manual Augmentation]
        C1 --> D1[Augmented Images]
        D1 --> E1[Manual Feature Extraction]
        E1 --> F1[features.h5]
    end
    
    subgraph "Recognition Flow"
        A2[Input Image] --> B2[feature_extractor.py]
        B2 --> C2[CLIP + DINOv2]
        C2 --> D2[1536D Features]
        D2 --> E2[FAISS Search]
        E2 --> F2[Recognition Result]
    end
    
    subgraph "Storage Flow"
        A3[Multiple Images] --> B3[Manual Scripts]
        B3 --> C3[Scattered Processing]
        C3 --> D3[Legacy Storage]
        D3 --> E3[Index Rebuilding]
    end
    
    style A1 fill:#ffcccc
    style A2 fill:#ffcccc
    style A3 fill:#ffcccc
    style F1 fill:#ccccff
    style F2 fill:#ccccff
    style E3 fill:#ccccff
```

### **UNIFIED SYSTEM** (Integrated Architecture)
```mermaid
graph TD
    subgraph "Unified Preprocessing Pipeline"
        A[Image Input] --> B[Input Manager]
        B --> C{Processing Mode}
        
        C -->|Training| D[Augmentation Engine]
        C -->|Inference| E[Feature Extractor]
        C -->|Batch| F[Batch Processor]
        
        D --> G[SQLite Storage]
        E --> G
        F --> G
        
        G --> H[FAISS Index Update]
        H --> I[Recognition Ready]
    end
    
    subgraph "Platform Optimization"
        J[Platform Detector] --> K[Optimization Config]
        K --> L[Memory Manager]
        L --> M[Cache Manager]
    end
    
    subgraph "Storage Layer"
        N[processed_images]
        O[feature_cache] 
        P[augmentation_batches]
        Q[augmented_items]
    end
    
    G --> N
    G --> O
    G --> P
    G --> Q
    
    K --> C
    L --> C
    M --> C
    
    style A fill:#ccffcc
    style I fill:#ccffff
    style G fill:#ffffcc
```

## Detailed Component Flow

### **1. Image Input Processing**
```mermaid
graph LR
    subgraph "Input Sources"
        A1[Camera Capture]
        A2[File Upload]
        A3[Batch Files]
        A4[URL Import]
    end
    
    subgraph "Input Manager"
        B1[Source Validation]
        B2[Format Standardization]
        B3[Metadata Extraction]
        B4[Hash Generation]
    end
    
    subgraph "Processed Image"
        C1[Normalized Array]
        C2[Processing Metadata]
        C3[Cache Key]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    B4 --> C1
    B4 --> C2
    B4 --> C3
```

### **2. Training Augmentation Flow**
```mermaid
graph TD
    A[Source Image] --> B[Augmentation Engine]
    
    subgraph "Strategy Selection"
        C1[Geometric 30%]
        C2[Perspective 25%]
        C3[Lighting 25%]
        C4[Noise/Blur 15%]
        C5[Effects 5%]
    end
    
    B --> C1
    B --> C2
    B --> C3
    B --> C4
    B --> C5
    
    subgraph "Batch Processing"
        D1[GPU Acceleration]
        D2[Memory Optimization]
        D3[Quality Control]
    end
    
    C1 --> D1
    C2 --> D1
    C3 --> D1
    C4 --> D2
    C5 --> D2
    
    D1 --> D3
    D2 --> D3
    
    subgraph "Storage Integration"
        E1[Feature Extraction]
        E2[SQLite Transaction]
        E3[Index Update]
    end
    
    D3 --> E1
    E1 --> E2
    E2 --> E3
    
    E3 --> F[400+ Training Images]
```

### **3. Feature Extraction Flow**
```mermaid
graph TD
    A[Processed Image] --> B{Cache Check}
    B -->|Hit| C[Return Cached Features]
    B -->|Miss| D[Platform Detection]
    
    D --> E{Platform Type}
    E -->|Apple Silicon| F1[MPS Acceleration]
    E -->|NVIDIA GPU| F2[CUDA Acceleration]
    E -->|CPU| F3[Multi-thread Processing]
    
    F1 --> G[Model Processing]
    F2 --> G
    F3 --> G
    
    subgraph "Dual Model Processing"
        H1[CLIP Processing]
        H2[DINOv2 Processing]
    end
    
    G --> H1
    G --> H2
    
    H1 --> I1[768D CLIP Features]
    H2 --> I2[768D DINOv2 Features]
    
    I1 --> J[Feature Combination]
    I2 --> J
    
    J --> K[1536D Combined Features]
    K --> L[L2 Normalization]
    L --> M[Cache Storage]
    M --> N[Return Features]
```

### **4. SQLite Integration Flow**
```mermaid
graph TD
    subgraph "Data Input"
        A1[Processed Image]
        A2[Feature Vector]
        A3[Metadata]
    end
    
    subgraph "SQLite Operations"
        B1[Transaction Start]
        B2[Data Validation]
        B3[Duplicate Check]
        B4[Insert Operations]
        B5[Index Updates]
        B6[Transaction Commit]
    end
    
    subgraph "Tables"
        C1[processed_images]
        C2[feature_cache]
        C3[augmentation_batches]
        C4[augmented_items]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> B6
    
    B4 --> C1
    B4 --> C2
    B4 --> C3
    B4 --> C4
    
    B6 --> D[FAISS Index Update]
    D --> E[Recognition Ready]
```

## Performance Optimization Flow

### **Memory Management**
```mermaid
graph LR
    A[Memory Monitor] --> B{Available Memory}
    B -->|High| C[Large Batches]
    B -->|Medium| D[Standard Batches]
    B -->|Low| E[Small Batches + Cleanup]
    
    C --> F[Batch Processing]
    D --> F
    E --> F
    
    F --> G[Memory Cleanup]
    G --> H{GPU Memory}
    H -->|CUDA| I[torch.cuda.empty_cache()]
    H -->|MPS| J[torch.mps.empty_cache()]
    H -->|CPU| K[gc.collect()]
    
    I --> L[Next Batch]
    J --> L
    K --> L
```

### **Caching Strategy**
```mermaid
graph TD
    A[Feature Request] --> B{Memory Cache}
    B -->|Hit| C[Return Cached]
    B -->|Miss| D{Disk Cache}
    D -->|Hit| E[Load to Memory]
    D -->|Miss| F[Compute Features]
    
    E --> G[Return Features]
    F --> H[Store in Caches]
    H --> G
    
    subgraph "Cache Management"
        I[LRU Eviction]
        J[Size Monitoring]
        K[TTL Cleanup]
    end
    
    G --> I
    I --> J
    J --> K
```

## Integration Points

### **Frontend Integration**
```mermaid
graph LR
    subgraph "Frontend UI"
        A1[Camera Widget]
        A2[File Upload Widget]
        A3[Batch Upload Widget]
    end
    
    subgraph "Unified Interface"
        B1[UnifiedFrontendInterface]
        B2[Processing Mode Selection]
        B3[Progress Tracking]
    end
    
    subgraph "Backend Processing"
        C1[UnifiedPreprocessingPipeline]
        C2[SQLite Storage]
        C3[Recognition Engine]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B1 --> B2
    B2 --> B3
    
    B3 --> C1
    C1 --> C2
    C2 --> C3
    
    C3 --> D[Results Display]
```

### **API Integration**
```mermaid
graph TD
    subgraph "REST API Endpoints"
        A1[/upload/single]
        A2[/upload/batch]
        A3[/process/augment]
        A4[/recognize/image]
    end
    
    subgraph "Unified Processing"
        B1[Request Validation]
        B2[Processing Mode Detection]
        B3[Pipeline Execution]
        B4[Response Formatting]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    B4 --> C[JSON Response]
```

## Error Handling Flow

### **Processing Pipeline Error Handling**
```mermaid
graph TD
    A[Image Processing] --> B{Processing Success?}
    B -->|Success| C[Continue Pipeline]
    B -->|Error| D[Error Classification]
    
    D --> E{Error Type}
    E -->|Corrupted Image| F[Skip with Warning]
    E -->|Memory Error| G[Reduce Batch Size]
    E -->|Model Error| H[Fallback Processing]
    E -->|Storage Error| I[Retry with Cleanup]
    
    F --> J[Log Warning]
    G --> K[Retry Processing]
    H --> L[CPU Fallback]
    I --> M[Clear Cache + Retry]
    
    J --> N[Continue with Next]
    K --> A
    L --> C
    M --> A
    
    C --> O[Success Result]
```

This comprehensive flow diagram shows how the unified preprocessing system will consolidate all image processing workflows into a single, efficient, platform-optimized pipeline with SQLite integration. The system provides clear separation of concerns while maintaining high performance and error resilience.