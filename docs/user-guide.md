# User Guide - AI Recognition System

## Getting Started

### Starting the System
```bash
# Single command to start everything
python3 start_system.py
```

This launches both the backend API server and the PyQt6 GUI interface.

## GUI Interface Overview

The system provides a professional desktop interface with the following tabs:

### 1. Dashboard Tab
- **System Status**: Real-time system health and performance metrics
- **Quick Stats**: Number of items, recognition accuracy, system uptime
- **Recent Activity**: Latest recognition results and system events

### 2. Items Tab - Item Management

#### Adding New Items
1. **Click "Add New Item"** button
2. **Fill Item Details**:
   - Item Name (required)
   - Category (optional)
   - Description (optional)
3. **Upload Images**:
   - **Drag and drop** images onto the upload area
   - Or **click "Browse"** to select files
   - **Minimum**: 8 high-quality images per item
   - **Recommended**: 10-15 images for best accuracy
4. **Save Item** to add to database

#### Image Requirements
- **Formats**: JPG, PNG, JPEG
- **Resolution**: 1080p or higher recommended
- **Variety**: Different angles, lighting, backgrounds
- **Quality**: Clear, well-lit, in-focus images

#### Managing Existing Items
- **View Items**: Browse all items in scrollable list
- **Edit Items**: Click item to modify details or add images
- **Delete Items**: Remove items (will require retraining)

### 3. Recognition Tab - Real-Time Recognition

#### Camera Recognition
1. **Connect Camera**: System automatically detects available cameras
2. **Start Recognition**: Click "Start Camera" for live recognition
3. **View Results**: Real-time item identification with confidence scores
4. **Save Results**: Option to save recognition results to file

#### File Upload Recognition
1. **Upload Images**: Drag and drop or browse for image files
2. **Batch Processing**: Upload multiple images for batch recognition
3. **View Results**: Detailed results with confidence scores and processing time
4. **Export Results**: Save results in CSV or JSON format

#### Recognition Results
- **Item ID**: Identified item name or "unknown"
- **Confidence Score**: 0.0-2.0 scale (higher is better)
- **Processing Time**: Recognition speed in milliseconds
- **Status**: Success, unknown item, or error

### 4. Training Tab - Model Training

#### Training Process
1. **Review Configuration**:
   - Training epochs (default: 40)
   - Batch size (default: 24)
   - Learning rate (default: 3e-4)
2. **Start Training**: Click "Start Training" button
3. **Monitor Progress**:
   - Real-time loss curves
   - Accuracy metrics
   - Estimated completion time
4. **Training Completion**: System automatically saves best model

#### When to Retrain
- **After adding new items** (always required)
- **When accuracy drops** below acceptable levels
- **After deleting items** to optimize performance

#### Training Configuration
```python
# Default training settings (can be modified in GUI)
epochs: 40           # Number of training iterations
batch_size: 24       # Images per batch
learning_rate: 3e-4  # Optimization speed
augmentations: 50    # Synthetic data per original image
```

### 5. Evaluation Tab - Performance Analysis

#### Running Evaluation
1. **Click "Run Evaluation"** to test system performance
2. **View Metrics**:
   - Overall accuracy percentage
   - Per-item accuracy breakdown
   - Confusion matrix
   - Processing speed statistics
3. **Export Report**: Save detailed evaluation report

#### Key Metrics
- **Recognition Accuracy**: Percentage of correct identifications
- **Unknown Detection**: Rate of proper unknown item rejection
- **Average Processing Time**: Speed per image
- **Confidence Distribution**: Range of confidence scores

### 6. Settings Tab - System Configuration

#### Recognition Settings
- **Confidence Threshold**: Minimum confidence for positive identification
- **Unknown Threshold**: Maximum confidence for unknown item detection
- **Processing Mode**: Fast vs. accurate recognition

#### System Settings
- **GPU Acceleration**: Enable/disable GPU processing
- **Batch Processing**: Configure batch sizes for memory optimization
- **Logging Level**: Control detail level of system logs

### 7. Logs Tab - System Monitoring

#### Log Viewer Features
- **Real-time Updates**: Live log streaming
- **Filtering**: Filter by log level (INFO, WARNING, ERROR)
- **Search**: Find specific log entries
- **Export**: Save logs to file for debugging

#### Log Categories
- **System Events**: Startup, shutdown, configuration changes
- **Recognition Events**: Each recognition attempt with details
- **Training Events**: Training progress, model saves, errors
- **API Events**: Backend requests and responses

## Workflow Examples

### Complete Workflow: Adding New Items

1. **Prepare Images**:
   ```
   - Take 8-15 high-quality photos of the item
   - Ensure variety in angles and lighting
   - Save in easily accessible folder
   ```

2. **Add Item via GUI**:
   ```
   Items Tab → Add New Item → Fill details → Upload images → Save
   ```

3. **Trigger Training**:
   ```
   Training Tab → Review settings → Start Training → Monitor progress
   ```

4. **Evaluate Performance**:
   ```
   Evaluation Tab → Run Evaluation → Review metrics
   ```

5. **Test Recognition**:
   ```
   Recognition Tab → Upload test image → Verify correct identification
   ```

### Daily Usage: Item Recognition

1. **Start System**:
   ```bash
   python3 start_system.py
   ```

2. **Recognition Methods**:
   - **Live Camera**: Recognition Tab → Start Camera
   - **File Upload**: Recognition Tab → Upload Images
   - **Batch Processing**: Upload multiple files simultaneously

3. **Review Results**:
   - Check confidence scores (>0.7 typically good)
   - Verify unknown items are properly rejected
   - Export results if needed for records

## Best Practices

### Image Collection
- **Consistent Quality**: Use same camera/lighting when possible
- **Multiple Angles**: Front, back, sides, angled views
- **Real Conditions**: Match actual usage environment
- **Background Variety**: Different backgrounds prevent overfitting

### Training Optimization
- **Regular Retraining**: After adding 5+ new items
- **Monitor Accuracy**: Retrain if accuracy drops below 95%
- **Backup Models**: Keep copies of well-performing models

### System Maintenance
- **Regular Evaluation**: Weekly performance checks
- **Log Monitoring**: Check for errors or warnings
- **Index Optimization**: Rebuild index periodically for large datasets

## Performance Expectations

### Accuracy Targets
- **Known Items**: 95%+ recognition accuracy
- **Unknown Items**: 100% proper rejection (no false positives)
- **Overall System**: 98%+ combined accuracy

### Speed Targets
- **Recognition Time**: <500ms per image
- **Training Time**: 1-4 hours depending on dataset size
- **System Startup**: <30 seconds to ready state

### Capacity Limits
- **Items Supported**: 1,000+ items per system
- **Images per Item**: 8-50 images recommended
- **Memory Usage**: <4GB for 1,000 items
- **Storage**: ~10GB for complete system with 1,000 items

## Troubleshooting Common Issues

### Recognition Problems
- **Low Confidence**: Add more training images, retrain
- **Wrong Recognition**: Check for similar-looking items, retrain
- **Unknown Items Recognized**: Increase confidence threshold

### GUI Issues
- **Frozen Interface**: Restart application, check system resources
- **Image Upload Fails**: Check file format, size limits
- **Training Doesn't Start**: Verify sufficient training data

### Performance Issues
- **Slow Recognition**: Enable GPU acceleration, reduce image resolution
- **High Memory Usage**: Reduce batch sizes in settings
- **Training Too Slow**: Use GPU, reduce augmentation count

## Advanced Features

### API Integration
The system provides a REST API for integration with other systems:

```bash
# Recognition API example
curl -X POST "http://localhost:8000/recognize" \
     -F "file=@image.jpg"
```

### Batch Processing
For large-scale operations:
- Use the batch upload feature in Recognition tab
- Process entire directories of images
- Export results in various formats

### Custom Configuration
Modify `config.yaml` for advanced customization:
- Model parameters
- Feature extraction settings
- Performance tuning options

## Getting Help

- **Built-in Help**: Hover over GUI elements for tooltips
- **Log Analysis**: Check Logs tab for detailed error information
- **Documentation**: Refer to technical documentation for advanced topics
- **Performance Monitoring**: Use Evaluation tab to track system health