# Demo Images for Enhanced WeakTR Web Interface

This directory contains sample MRI images for testing the web interface.

## 📁 Sample Images

### BraTS Dataset Samples
- `sample_1.jpg` - Sample brain MRI with tumor regions
- `sample_2.jpg` - Another sample for testing
- `sample_3.jpg` - Additional test case

## 🎯 How to Use Demo Images

1. **Start the web interface**:
   ```bash
   python run_web_interface.py
   ```

2. **Upload demo images**:
   - Drag and drop any image from this folder to the upload area
   - Or click "Choose File" and select an image

3. **View results**:
   - The interface will process the image and show segmentation results
   - Compare different images to see how the model performs

## 📊 Expected Results

Based on the trained model (72.66% mIoU), you should see:

- **Clear segmentation boundaries** between different tissue types
- **Accurate tumor region identification** (blue areas)
- **Proper edema detection** (green areas)
- **Necrotic tissue identification** (red areas)
- **Background tissue preservation** (black areas)

## 🔍 Testing Tips

1. **Try different image types**:
   - T1-weighted MRI
   - T2-weighted MRI
   - FLAIR sequences
   - Contrast-enhanced images

2. **Check metrics**:
   - Area percentages for each class
   - Confidence scores
   - Overall segmentation quality

3. **Compare results**:
   - Original vs. segmentation
   - Overlay visualization
   - Probability heatmaps

## ⚠️ Important Notes

- **Image quality matters**: Higher resolution images generally give better results
- **Preprocessing**: Images are automatically resized to 256x256 pixels
- **Format support**: JPG, PNG, JPEG formats are supported
- **File size**: Maximum 16MB per image

## 🐛 Troubleshooting

If demo images don't work:

1. **Check file format**: Ensure images are JPG, PNG, or JPEG
2. **Verify file size**: Must be under 16MB
3. **Check model loading**: Ensure `best_model.pth` is available
4. **GPU memory**: Large images may require more GPU memory

## 📈 Performance Expectations

With the trained model:
- **Inference time**: 2-3 seconds per image
- **Accuracy**: 72.66% mIoU on BraTS dataset
- **Memory usage**: ~4-8GB GPU memory recommended

---

**Note**: These are sample images for testing purposes. For clinical use, ensure you have proper medical image data and follow relevant regulations. 