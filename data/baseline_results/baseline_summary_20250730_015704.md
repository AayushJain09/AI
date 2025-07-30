# Performance Baseline Report

**Generated:** 2025-07-30T01:57:04.335843
**Platform:** Apple_Silicon

## System Information

- **CPU:** 10 cores @ 3504 MHz
- **Memory:** 16.0 GB total, 7.0 GB available
- **GPU:** Available
- **Architecture:** unknown

## Performance Summary

### Vector Insert Small

- **Success Rate:** 100.0%
- **Average Duration:** 10.49ms
- **95th Percentile:** 12.19ms
- **Peak Memory:** 450.8MB

### Vector Insert Medium

- **Success Rate:** 100.0%
- **Average Duration:** 9.57ms
- **95th Percentile:** 10.27ms
- **Peak Memory:** 452.3MB

### Vector Insert Large

- **Success Rate:** 100.0%
- **Average Duration:** 11.23ms
- **95th Percentile:** 12.13ms
- **Peak Memory:** 454.8MB

### Metadata Insert Basic

- **Success Rate:** 100.0%
- **Average Duration:** 0.29ms
- **95th Percentile:** 0.31ms
- **Peak Memory:** 455.8MB

### Metadata Insert Complex

- **Success Rate:** 100.0%
- **Average Duration:** 0.27ms
- **95th Percentile:** 0.28ms
- **Peak Memory:** 455.8MB

### Metadata Retrieval

- **Success Rate:** 100.0%
- **Average Duration:** 0.74ms
- **95th Percentile:** 0.81ms
- **Peak Memory:** 456.6MB

### Analytics Simple Count

- **Success Rate:** 100.0%
- **Average Duration:** 0.63ms
- **95th Percentile:** 0.74ms
- **Peak Memory:** 469.3MB

### Analytics Aggregation

- **Success Rate:** 100.0%
- **Average Duration:** 1.71ms
- **95th Percentile:** 3.69ms
- **Peak Memory:** 469.5MB

### Analytics Time Series

- **Success Rate:** 100.0%
- **Average Duration:** 1.57ms
- **95th Percentile:** 3.18ms
- **Peak Memory:** 469.9MB

## Optimization Recommendations

- ✅ Apple Silicon detected - unified memory architecture optimized
- 💡 Consider increasing batch sizes for vector operations
- ⚠️ Limited memory available - reduce cache sizes
- ⚠️ vector_search has low success rate: 0.0%
- ⚠️ recognition_pipeline has low success rate: 0.0%
- ⚠️ concurrent_operation has low success rate: 0.0%
- 📊 Monitor performance metrics regularly
- 🔄 Run baseline periodically to detect performance regression
- ⚡ Consider enabling query result caching for analytics
- 📈 Scale resources based on actual usage patterns

## Resource Usage

- **Peak Memory Usage:** 469.4MB
- **System Memory Usage:** 56.6%
