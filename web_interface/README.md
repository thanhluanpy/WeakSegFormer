# Enhanced WeakTR Web Interface

## 🧠 Advanced Brain Tumor Segmentation Web Application

A modern, user-friendly web interface for the Enhanced WeakTR model that achieves **72.66% mIoU** performance on brain tumor segmentation.

## ✨ Features

- **🎯 High Performance**: 72.66% mIoU achieved on BraTS dataset
- **🖼️ Interactive Upload**: Drag & drop or click to upload MRI images
- **📊 Real-time Analysis**: Instant segmentation results with detailed metrics
- **🎨 Beautiful Visualization**: Multiple views (Original, Segmentation, Overlay)
- **📈 Detailed Metrics**: Area percentages and confidence scores for each class
- **💾 Download Results**: Save visualization and analysis data
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd web_interface
pip install -r requirements.txt
```

### 2. Prepare Model Files

Ensure you have the trained model files in the correct location:
- `advanced_results/best_model.pth` - Trained model weights
- `advanced_results/args.json` - Model configuration

### 3. Run the Application

```bash
python app.py
```

### 4. Access the Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 📋 System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible GPU (recommended for faster inference)
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 2GB+ free space

## 🎯 Supported Image Formats

- **Formats**: JPG, JPEG, PNG
- **Max Size**: 16MB
- **Resolution**: Any (automatically resized to 256x256)

## 🏷️ Segmentation Classes

The model segments brain MRI images into 4 classes:

| Class | Color | Description |
|-------|-------|-------------|
| Background | Black | Normal brain tissue |
| Necrotic | Red | Dead tumor tissue |
| Edema | Green | Swelling around tumor |
| Tumor | Blue | Active tumor tissue |

## 📊 Output Metrics

For each uploaded image, the system provides:

### Area Analysis
- **Percentage coverage** for each class
- **Pixel count** analysis
- **Relative distribution** across classes

### Confidence Scores
- **Average confidence** for each class prediction
- **Probability heatmaps** for detailed analysis
- **Uncertainty quantification**

## 🎨 Visualization Features

### 1. Original Image
- Input MRI image after preprocessing
- 256x256 resolution for model compatibility

### 2. Segmentation Map
- Color-coded segmentation results
- Clear class boundaries and regions

### 3. Overlay Result
- Original image with segmentation overlay
- 70% original + 30% segmentation for clarity

### 4. Probability Heatmaps
- Individual class probability maps
- Confidence visualization for each region

## 💾 Download Options

### Visualization Download
- High-resolution PNG file (300 DPI)
- Complete analysis visualization
- Professional quality for reports

### Results Download
- JSON file with all metrics
- Base64 encoded images
- Timestamp and analysis metadata

## 🔧 API Endpoints

### Health Check
```
GET /health
```
Returns system status and model loading information.

### File Upload
```
POST /upload
```
Accepts image file and returns segmentation results.

### Results Retrieval
```
GET /results/<filename>
```
Retrieve saved results by filename.

## 🛠️ Customization

### Model Configuration
Edit `app.py` to modify:
- Model loading parameters
- Preprocessing settings
- Output visualization options

### Styling
Modify `templates/index.html` to customize:
- Color scheme
- Layout design
- User interface elements

### Performance Tuning
Adjust in `app.py`:
- Batch size for inference
- Image preprocessing parameters
- Memory optimization settings

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure `best_model.pth` exists in `advanced_results/`
   - Check GPU memory availability
   - Verify CUDA installation

2. **Memory Issues**
   - Reduce batch size in app.py
   - Use CPU-only mode if GPU memory is insufficient
   - Close other applications

3. **Upload Errors**
   - Check file format (JPG, PNG, JPEG only)
   - Ensure file size < 16MB
   - Verify file is not corrupted

### Performance Optimization

1. **GPU Acceleration**
   - Install CUDA toolkit
   - Use compatible PyTorch version
   - Monitor GPU memory usage

2. **Batch Processing**
   - Implement queue system for multiple uploads
   - Add progress indicators
   - Optimize image preprocessing

## 📈 Performance Metrics

Based on the trained model:

| Metric | Value | Target |
|--------|-------|--------|
| mIoU | 72.66% | >50% ✅ |
| Dice Score | ~85% | >65% ✅ |
| Training Epochs | 100 | 60-100 ✅ |
| Inference Time | ~2-3s | <5s ✅ |

## 🔮 Future Enhancements

- [ ] Batch processing support
- [ ] Real-time video analysis
- [ ] 3D volume visualization
- [ ] Integration with PACS systems
- [ ] Mobile app development
- [ ] Cloud deployment options

## 📞 Support

For technical support or questions:
- Check the troubleshooting section
- Review model training logs
- Verify system requirements

## 📄 License

This project is part of the Enhanced WeakTR research implementation.

---

**Enhanced WeakTR Web Interface** - Making advanced brain tumor segmentation accessible to everyone! 🧠✨ 