# Web Interface Debug Guide

## Overview
This guide helps you troubleshoot issues with the Enhanced WeakTR Brain Tumor Segmentation web interface.

## Quick Start
1. **Start the web interface**: `python app.py`
2. **Open browser**: Navigate to `http://localhost:5000`
3. **Check health**: Look for the health status message at the top

## Common Issues and Solutions

### 1. "An error occurred during processing" Error

**Symptoms**: 
- Model loads successfully (green checkmark in debug report)
- Test inference works
- But upload fails with generic error message

**Causes**:
- Postprocessing issues with model output
- Visualization creation failures
- Metrics calculation problems
- Base64 conversion errors

**Solutions**:
1. **Check server logs** for detailed error messages
2. **Use Debug Model button** to verify model status
3. **Check file permissions** for uploads/ and results/ directories
4. **Verify model checkpoint** exists in `advanced_results/best_model.pth`

### 2. Model Loading Issues

**Symptoms**:
- Health check shows "Model not loaded"
- Debug Model shows "No Model" status

**Solutions**:
1. **Check model files**:
   ```
   advanced_results/
   ├── best_model.pth (should exist, ~309MB)
   └── args.json (should exist, ~1KB)
   ```

2. **Verify CUDA availability**:
   - Check if PyTorch CUDA is working
   - Ensure GPU drivers are up to date

3. **Check Python dependencies**:
   ```bash
   pip install torch torchvision
   pip install albumentations
   pip install pillow opencv-python
   ```

### 3. Performance Issues

**Symptoms**:
- UI loads slowly
- Processing takes a long time
- Memory usage is high

**Solutions**:
1. **Use GPU acceleration** (if available)
2. **Reduce image size** (currently supports up to 16MB)
3. **Check system resources** (RAM, GPU memory)

## Debug Tools

### 1. Health Check Endpoint
**URL**: `GET /health`
**Purpose**: Overall system health status
**Response**: JSON with system status, model status, dependencies

### 2. Debug Model Endpoint
**URL**: `GET /debug/model`
**Purpose**: Detailed model testing and diagnostics
**Response**: JSON with model status, test inference results, filesystem info

### 3. Debug Model Button
**Location**: Web interface (blue button with bug icon)
**Purpose**: User-friendly way to test model functionality
**Action**: Runs test inference and displays detailed report

## Testing the System

### Manual Testing
1. **Health Check**: Visit `/health` endpoint
2. **Debug Model**: Click "Debug Model" button
3. **File Upload**: Try uploading a small test image

### Automated Testing
Run the test script:
```bash
cd web_interface
python test_simple_upload.py
```

This script will:
- Test health check endpoint
- Test debug model endpoint
- Test file upload functionality
- Provide detailed feedback

## Log Analysis

### Key Log Messages to Look For

**✅ Success Indicators**:
```
✅ Model loaded successfully! Best mIoU: 72.66
✅ Postprocessing completed
✅ Visualization created
✅ Metrics calculated
✅ All images converted to base64
✅ Results prepared
✅ File upload processing completed successfully!
```

**❌ Error Indicators**:
```
❌ Error loading model: [error details]
❌ Postprocessing failed: [error details]
❌ Visualization creation failed: [error details]
❌ Metrics calculation failed: [error details]
❌ Image conversion to base64 failed: [error details]
```

### Debug Mode
Enable verbose logging by setting environment variable:
```bash
export FLASK_ENV=development
python app.py
```

## File Structure Requirements

```
web_interface/
├── app.py (main application)
├── uploads/ (upload directory - must be writable)
├── results/ (results directory - must be writable)
└── templates/
    └── index.html (frontend template)

../ (parent directory)
├── models_enhanced.py (model definitions)
├── advanced_results/
│   ├── best_model.pth (trained model - required)
│   └── args.json (model configuration - required)
└── datasets.py (dataset utilities)
```

## Troubleshooting Steps

### Step 1: Basic Health Check
1. Start the web interface
2. Check browser console for errors
3. Look for health status message
4. Note any error messages

### Step 2: Model Verification
1. Click "Debug Model" button
2. Check if model loads successfully
3. Verify test inference works
4. Note model type and device

### Step 3: File System Check
1. Verify uploads/ directory exists and is writable
2. Verify results/ directory exists and is writable
3. Check file permissions

### Step 4: Dependency Verification
1. Check PyTorch installation
2. Verify CUDA availability (if using GPU)
3. Check other required packages

### Step 5: Model File Verification
1. Ensure `best_model.pth` exists in `advanced_results/`
2. Verify file size (~309MB)
3. Check `args.json` configuration

## Getting Help

If you continue to experience issues:

1. **Check server logs** for detailed error messages
2. **Run test script** and share output
3. **Use Debug Model button** and share report
4. **Check browser console** for frontend errors
5. **Verify system requirements** (Python version, dependencies)

## System Requirements

- **Python**: 3.7+
- **PyTorch**: 1.8+
- **Memory**: 4GB+ RAM
- **Storage**: 1GB+ free space
- **GPU**: Optional but recommended for performance
- **OS**: Windows 10+, macOS 10.14+, or Linux 