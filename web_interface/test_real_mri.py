#!/usr/bin/env python3
"""
Test with real MRI image from specified path
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_real_mri():
    """Test with real MRI image"""
    print("🔍 Testing with Real MRI Image")
    print("=" * 60)
    
    # MRI image path
    mri_path = r"F:\MRI-Result\BraTS-GLI-00124-000_78.jpg"
    
    try:
        # Check if file exists
        if not os.path.exists(mri_path):
            print(f"❌ File not found: {mri_path}")
            return False
        
        print(f"✅ Found MRI file: {mri_path}")
        
        # Import the model loading function
        from app import load_model_global, preprocess_image, postprocess_output
        
        # Load model
        print("🔄 Loading model...")
        if not load_model_global():
            print("❌ Failed to load model")
            return False
        
        print("✅ Model loaded successfully")
        
        # Load and display original image
        print("🔄 Loading original MRI image...")
        original_image = np.array(Image.open(mri_path))
        print(f"✅ Original image shape: {original_image.shape}")
        print(f"✅ Original image dtype: {original_image.dtype}")
        print(f"✅ Original image range: [{original_image.min()}, {original_image.max()}]")
        
        # Preprocess image
        print("🔄 Preprocessing image...")
        image_tensor, processed_image = preprocess_image(mri_path)
        print(f"✅ Processed image shape: {processed_image.shape}")
        
        # Get model
        from app import model, device
        model.eval()
        
        # Run inference
        print("🔄 Running model inference...")
        with torch.no_grad():
            output = model(image_tensor.to(device))
            
            if isinstance(output, (list, tuple)):
                main_output = output[0]  # Use first output
            else:
                main_output = output
            
            print(f"✅ Model output shape: {main_output.shape}")
        
        # Apply softmax to get probabilities
        probs = torch.softmax(main_output, dim=1)
        probs_np = probs.cpu().numpy().squeeze()
        
        # Get predicted classes
        pred_classes = np.argmax(probs_np, axis=0)
        
        print(f"✅ Predicted classes shape: {pred_classes.shape}")
        print(f"✅ Unique predicted classes: {np.unique(pred_classes)}")
        
        # Analyze each class
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        print(f"\n📊 Class Analysis for Real MRI:")
        for i, class_name in enumerate(class_names):
            if i < probs_np.shape[0]:
                class_prob = probs_np[i]
                class_pixels = np.sum(pred_classes == i)
                total_pixels = pred_classes.size
                percentage = (class_pixels / total_pixels) * 100
                avg_prob = np.mean(class_prob)
                max_prob = np.max(class_prob)
                
                print(f"   - {class_name} (Class {i}):")
                print(f"     * Pixels: {class_pixels:,}/{total_pixels:,} ({percentage:.1f}%)")
                print(f"     * Avg probability: {avg_prob:.4f}")
                print(f"     * Max probability: {max_prob:.4f}")
        
        # Create comprehensive visualization
        print("\n🎨 Creating comprehensive visualization...")
        create_comprehensive_visualization(original_image, processed_image, pred_classes, probs_np, class_names)
        
        # Test postprocessing
        print("\n🔄 Testing postprocessing...")
        try:
            segmentation_map, colored_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors, individual_class_maps = postprocess_output(
                main_output, processed_image
            )
            
            print(f"✅ Postprocessing successful:")
            print(f"   - Segmentation map shape: {segmentation_map.shape}")
            print(f"   - Colored map shape: {colored_map.shape}")
            print(f"   - Present classes: {present_classes}")
            print(f"   - Actual class names: {actual_class_names}")
            print(f"   - Individual class maps: {list(individual_class_maps.keys())}")
            
            # Create postprocessing visualization
            create_postprocessing_visualization(original_image, processed_image, segmentation_map, colored_map, overlay, individual_class_maps, class_names)
            
        except Exception as e:
            print(f"❌ Postprocessing failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Real MRI testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in real MRI testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comprehensive_visualization(original_image, processed_image, pred_classes, probs_np, class_names):
    """Create comprehensive visualization of real MRI results"""
    try:
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Real MRI Segmentation Analysis', fontsize=18, fontweight='bold')
        
        # Row 1: Original and processed images
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original MRI Image', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(processed_image)
        axes[0, 1].set_title('Processed Image', fontweight='bold', fontsize=14)
        axes[0, 1].axis('off')
        
        # Raw predicted classes
        im1 = axes[0, 2].imshow(pred_classes, cmap='tab10')
        axes[0, 2].set_title('Raw Predicted Classes', fontweight='bold', fontsize=14)
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Class distribution
        unique_classes, class_counts = np.unique(pred_classes, return_counts=True)
        colors = ['black', 'darkred', 'green', 'blue']
        
        bars = axes[0, 3].bar(unique_classes, class_counts, color=[colors[i] for i in unique_classes])
        axes[0, 3].set_title('Class Distribution', fontweight='bold', fontsize=14)
        axes[0, 3].set_xlabel('Class Index')
        axes[0, 3].set_ylabel('Pixel Count')
        axes[0, 3].set_xticks(unique_classes)
        axes[0, 3].set_xticklabels([class_names[i] for i in unique_classes])
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            percentage = (count / pred_classes.size) * 100
            axes[0, 3].text(bar.get_x() + bar.get_width()/2., height + 100,
                           f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        # Row 2: Probability maps for each class
        for i, class_name in enumerate(class_names):
            if i < probs_np.shape[0]:
                im = axes[1, i].imshow(probs_np[i], cmap='hot', vmin=0, vmax=1)
                axes[1, i].set_title(f'{class_name} Probability Map', fontweight='bold', fontsize=14)
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            else:
                axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                              transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{class_name} Probability Map', fontweight='bold', fontsize=14)
                axes[1, i].axis('off')
        
        # Row 3: Binary masks for each class
        for i, class_name in enumerate(class_names):
            if i < 4:
                # Create binary mask for this class
                class_mask = (pred_classes == i).astype(np.uint8)
                
                if np.any(class_mask):
                    axes[2, i].imshow(class_mask, cmap='gray')
                    axes[2, i].set_title(f'{class_name} Binary Mask', fontweight='bold', fontsize=14)
                    axes[2, i].axis('off')
                else:
                    axes[2, i].text(0.5, 0.5, 'No pixels', ha='center', va='center', 
                                  transform=axes[2, i].transAxes)
                    axes[2, i].set_title(f'{class_name} Binary Mask', fontweight='bold', fontsize=14)
                    axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = "real_mri_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating comprehensive visualization: {e}")

def create_postprocessing_visualization(original_image, processed_image, segmentation_map, colored_map, overlay, individual_class_maps, class_names):
    """Create postprocessing visualization"""
    try:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Postprocessing Results for Real MRI', fontsize=16, fontweight='bold')
        
        # Row 1: Original images and segmentation
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original MRI', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(processed_image)
        axes[0, 1].set_title('Processed Image', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(segmentation_map, cmap='tab10')
        axes[0, 2].set_title('Segmentation Map', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(colored_map)
        axes[0, 3].set_title('Colored Map', fontweight='bold')
        axes[0, 3].axis('off')
        
        # Row 2: Overlay and individual class maps
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Show individual class maps
        for i, class_name in enumerate(class_names[:3]):
            if class_name in individual_class_maps:
                axes[1, 1 + i].imshow(individual_class_maps[class_name])
                axes[1, 1 + i].set_title(f'{class_name} Individual', fontweight='bold')
                axes[1, 1 + i].axis('off')
            else:
                axes[1, 1 + i].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                  transform=axes[1, 1 + i].transAxes)
                axes[1, 1 + i].set_title(f'{class_name} Individual', fontweight='bold')
                axes[1, 1 + i].axis('off')
        
        plt.tight_layout()
        
        # Save postprocessing visualization
        postprocessing_path = "real_mri_postprocessing.png"
        plt.savefig(postprocessing_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Postprocessing visualization saved to: {postprocessing_path}")
        
    except Exception as e:
        print(f"❌ Error creating postprocessing visualization: {e}")

if __name__ == '__main__':
    print("🚀 Starting Real MRI Testing")
    print("=" * 60)
    
    success = test_real_mri()
    
    if success:
        print("\n🎉 Real MRI testing completed successfully!")
        print("✅ Check the generated images to see results with real MRI data")
    else:
        print("\n💥 Real MRI testing failed!")
        print("⚠️ Check the error messages above for details")