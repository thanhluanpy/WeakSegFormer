#!/usr/bin/env python3
"""
Show model predictions clearly
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

def show_model_predictions():
    """Show clear model predictions"""
    print("🔍 Showing Model Predictions")
    print("=" * 50)
    
    try:
        # Import the model loading function
        from app import load_model_global, preprocess_image
        
        # Load model
        print("🔄 Loading model...")
        if not load_model_global():
            print("❌ Failed to load model")
            return False
        
        print("✅ Model loaded successfully")
        
        # Use existing test image
        test_image_path = "test_mri.jpg"
        if not os.path.exists(test_image_path):
            print("❌ Test image not found. Run debug_model_output.py first.")
            return False
        
        # Preprocess image
        print("🔄 Preprocessing image...")
        image_tensor, original_image = preprocess_image(test_image_path)
        
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
        
        # Create clear visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Predictions - Raw Output Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original MRI Image', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')
        
        # Raw predicted classes
        im1 = axes[0, 1].imshow(pred_classes, cmap='tab10')
        axes[0, 1].set_title('Raw Predicted Classes\n(0=Background, 1=Necrotic, 2=Edema, 3=Tumor)', 
                            fontweight='bold', fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Class distribution
        unique_classes, class_counts = np.unique(pred_classes, return_counts=True)
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        colors = ['black', 'darkred', 'green', 'blue']
        
        bars = axes[0, 2].bar(unique_classes, class_counts, color=[colors[i] for i in unique_classes])
        axes[0, 2].set_title('Class Distribution', fontweight='bold', fontsize=14)
        axes[0, 2].set_xlabel('Class Index')
        axes[0, 2].set_ylabel('Pixel Count')
        axes[0, 2].set_xticks(unique_classes)
        axes[0, 2].set_xticklabels([class_names[i] for i in unique_classes])
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            percentage = (count / pred_classes.size) * 100
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 100,
                           f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        # Show probability maps for each class
        for i, class_name in enumerate(class_names):
            if i < 3:  # Show first 3 classes
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
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = "model_predictions_clear.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Clear predictions saved to: {output_path}")
        
        # Print detailed statistics
        print(f"\n📊 Detailed Statistics:")
        print(f"   - Image size: {original_image.shape}")
        print(f"   - Total pixels: {pred_classes.size}")
        print(f"   - Model output shape: {main_output.shape}")
        print(f"   - Probability range: [{probs_np.min():.4f}, {probs_np.max():.4f}]")
        
        print(f"\n🎨 Class Distribution:")
        for i, class_name in enumerate(class_names):
            if i < len(unique_classes):
                class_idx = unique_classes[i]
                count = class_counts[i]
                percentage = (count / pred_classes.size) * 100
                avg_prob = np.mean(probs_np[i])
                max_prob = np.max(probs_np[i])
                
                print(f"   - {class_name} (Class {class_idx}):")
                print(f"     * Pixels: {count:,} ({percentage:.1f}%)")
                print(f"     * Avg probability: {avg_prob:.4f}")
                print(f"     * Max probability: {max_prob:.4f}")
        
        # Create binary masks for each class
        print(f"\n🔍 Creating binary masks for each class...")
        create_binary_masks(pred_classes, probs_np, class_names, original_image)
        
        print(f"\n✅ Model predictions analysis completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in showing model predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_binary_masks(pred_classes, probs_np, class_names, original_image):
    """Create binary masks for each class"""
    try:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Binary Masks for Each Class', fontsize=16, fontweight='bold')
        
        # Show original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Show predicted classes
        axes[0, 1].imshow(pred_classes, cmap='tab10')
        axes[0, 1].set_title('All Predicted Classes', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Show individual class masks (only first 2 classes in this row)
        for i, class_name in enumerate(class_names[:2]):
            # Create binary mask for this class
            class_mask = (pred_classes == i).astype(np.uint8)
            
            if np.any(class_mask):
                # Show binary mask
                axes[0, 2 + i].imshow(class_mask, cmap='gray')
                axes[0, 2 + i].set_title(f'{class_name} Binary Mask', fontweight='bold')
                axes[0, 2 + i].axis('off')
                
                # Show probability map
                if i < probs_np.shape[0]:
                    im = axes[1, i].imshow(probs_np[i], cmap='hot', vmin=0, vmax=1)
                    axes[1, i].set_title(f'{class_name} Probability', fontweight='bold')
                    axes[1, i].axis('off')
                    plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
                else:
                    axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                  transform=axes[1, i].transAxes)
                    axes[1, i].set_title(f'{class_name} Probability', fontweight='bold')
                    axes[1, i].axis('off')
            else:
                axes[0, 2 + i].text(0.5, 0.5, 'No pixels', ha='center', va='center', 
                                  transform=axes[0, 2 + i].transAxes)
                axes[0, 2 + i].set_title(f'{class_name} Binary Mask', fontweight='bold')
                axes[0, 2 + i].axis('off')
                
                axes[1, i].text(0.5, 0.5, 'No pixels', ha='center', va='center', 
                              transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{class_name} Probability', fontweight='bold')
                axes[1, i].axis('off')
        
        # Show remaining classes in second row
        for i, class_name in enumerate(class_names[2:4]):
            class_idx = i + 2
            # Create binary mask for this class
            class_mask = (pred_classes == class_idx).astype(np.uint8)
            
            if np.any(class_mask):
                # Show binary mask
                axes[1, 2 + i].imshow(class_mask, cmap='gray')
                axes[1, 2 + i].set_title(f'{class_name} Binary Mask', fontweight='bold')
                axes[1, 2 + i].axis('off')
            else:
                axes[1, 2 + i].text(0.5, 0.5, 'No pixels', ha='center', va='center', 
                                  transform=axes[1, 2 + i].transAxes)
                axes[1, 2 + i].set_title(f'{class_name} Binary Mask', fontweight='bold')
                axes[1, 2 + i].axis('off')
        
        plt.tight_layout()
        
        # Save binary masks
        masks_path = "binary_masks_analysis.png"
        plt.savefig(masks_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Binary masks saved to: {masks_path}")
        
    except Exception as e:
        print(f"❌ Error creating binary masks: {e}")

if __name__ == '__main__':
    print("🚀 Starting Model Predictions Display")
    print("=" * 60)
    
    success = show_model_predictions()
    
    if success:
        print("\n🎉 Model predictions displayed successfully!")
        print("✅ Check the generated images to see what the model actually predicts")
    else:
        print("\n💥 Failed to display model predictions!")
        print("⚠️ Check the error messages above for details")