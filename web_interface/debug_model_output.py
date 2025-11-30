#!/usr/bin/env python3
"""
Debug script to show raw model output and compare with visualization
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def debug_model_output():
    """Debug and display raw model output"""
    print("🔍 Debugging Model Output - Raw Predictions")
    print("=" * 60)
    
    try:
        # Import the model loading function
        from app import load_model_global, preprocess_image, postprocess_output
        
        # Load model
        print("🔄 Loading model...")
        if not load_model_global():
            print("❌ Failed to load model")
            return False
        
        print("✅ Model loaded successfully")
        
        # Create a test image or use existing one
        print("📸 Creating test image...")
        
        # Option 1: Create synthetic test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Option 2: Use existing test image if available
        test_image_path = "test_mri.jpg"
        if os.path.exists(test_image_path):
            test_image = np.array(Image.open(test_image_path))
            print(f"✅ Using existing test image: {test_image_path}")
        else:
            # Save synthetic image for reference
            Image.fromarray(test_image).save(test_image_path)
            print(f"✅ Created synthetic test image: {test_image_path}")
        
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
            
            print(f"✅ Model output type: {type(output)}")
            print(f"✅ Model output length: {len(output) if isinstance(output, (list, tuple)) else 'Single tensor'}")
            
            if isinstance(output, (list, tuple)):
                print(f"✅ Output components:")
                for i, out in enumerate(output):
                    print(f"   - Output {i}: shape {out.shape}, range [{out.min().item():.4f}, {out.max().item():.4f}]")
                
                # Use the main output (usually the first one)
                main_output = output[0]
            else:
                main_output = output
                print(f"✅ Single output: shape {main_output.shape}, range [{main_output.min().item():.4f}, {main_output.max().item():.4f}]")
        
        # Apply softmax to get probabilities
        print("🔄 Applying softmax to get probabilities...")
        probs = torch.softmax(main_output, dim=1)
        probs_np = probs.cpu().numpy().squeeze()
        
        print(f"✅ Probabilities shape: {probs_np.shape}")
        print(f"✅ Probabilities range: [{probs_np.min():.4f}, {probs_np.max():.4f}]")
        
        # Get predicted classes
        pred_classes = np.argmax(probs_np, axis=0)
        print(f"✅ Predicted classes shape: {pred_classes.shape}")
        print(f"✅ Predicted classes range: [{pred_classes.min()}, {pred_classes.max()}]")
        print(f"✅ Unique predicted classes: {np.unique(pred_classes)}")
        
        # Analyze each class
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        print(f"\n📊 Class Analysis:")
        for i, class_name in enumerate(class_names):
            if i < probs_np.shape[0]:
                class_prob = probs_np[i]
                class_pixels = np.sum(pred_classes == i)
                total_pixels = pred_classes.size
                percentage = (class_pixels / total_pixels) * 100
                avg_prob = np.mean(class_prob)
                max_prob = np.max(class_prob)
                
                print(f"   - {class_name}:")
                print(f"     * Pixels: {class_pixels}/{total_pixels} ({percentage:.1f}%)")
                print(f"     * Avg probability: {avg_prob:.4f}")
                print(f"     * Max probability: {max_prob:.4f}")
                print(f"     * Probability range: [{np.min(class_prob):.4f}, {np.max(class_prob):.4f}]")
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        
        # 1. Raw probability maps
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Raw Model Output Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Predicted classes
        axes[0, 1].imshow(pred_classes, cmap='tab10')
        axes[0, 1].set_title('Predicted Classes', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Class probability maps (only show first 2 classes in this row)
        for i, class_name in enumerate(class_names[:2]):
            if i < probs_np.shape[0]:
                im = axes[0, 2 + i].imshow(probs_np[i], cmap='hot', vmin=0, vmax=1)
                axes[0, 2 + i].set_title(f'{class_name} Probability', fontweight='bold')
                axes[0, 2 + i].axis('off')
                plt.colorbar(im, ax=axes[0, 2 + i], fraction=0.046, pad=0.04)
            else:
                axes[0, 2 + i].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                  transform=axes[0, 2 + i].transAxes)
                axes[0, 2 + i].set_title(f'{class_name} Probability', fontweight='bold')
                axes[0, 2 + i].axis('off')
        
        # 2. Processed results
        print("🔄 Processing with postprocess_output...")
        try:
            segmentation_map, colored_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors, individual_class_maps = postprocess_output(
                main_output, original_image
            )
            
            print(f"✅ Postprocessing successful:")
            print(f"   - Segmentation map shape: {segmentation_map.shape}")
            print(f"   - Colored map shape: {colored_map.shape}")
            print(f"   - Present classes: {present_classes}")
            print(f"   - Actual class names: {actual_class_names}")
            print(f"   - Individual class maps: {list(individual_class_maps.keys())}")
            
            # Show processed results
            axes[1, 0].imshow(segmentation_map, cmap='tab10')
            axes[1, 0].set_title('Processed Segmentation', fontweight='bold')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(colored_map)
            axes[1, 1].set_title('Colored Map', fontweight='bold')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Overlay', fontweight='bold')
            axes[1, 2].axis('off')
            
            # Show individual class maps
            if individual_class_maps:
                first_class = list(individual_class_maps.keys())[0]
                axes[1, 3].imshow(individual_class_maps[first_class])
                axes[1, 3].set_title(f'{first_class} Individual Map', fontweight='bold')
                axes[1, 3].axis('off')
            else:
                axes[1, 3].text(0.5, 0.5, 'No Individual Maps', ha='center', va='center', 
                              transform=axes[1, 3].transAxes)
                axes[1, 3].set_title('Individual Maps', fontweight='bold')
                axes[1, 3].axis('off')
            
        except Exception as e:
            print(f"❌ Postprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error in the plot
            for i in range(4):
                axes[1, i].text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', 
                              transform=axes[1, i].transAxes, fontsize=8)
                axes[1, i].set_title(f'Error in {["Segmentation", "Colored Map", "Overlay", "Individual"][i]}', fontweight='bold')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save the analysis
        output_path = "model_output_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Analysis saved to: {output_path}")
        
        # Create detailed comparison
        print("\n🔍 Creating detailed comparison...")
        create_detailed_comparison(original_image, pred_classes, probs_np, class_names)
        
        print("\n✅ Model output debugging completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in model output debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_detailed_comparison(original_image, pred_classes, probs_np, class_names):
    """Create detailed comparison of raw output vs processed output"""
    try:
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Detailed Model Output Comparison', fontsize=18, fontweight='bold')
        
        # Row 1: Raw model output
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pred_classes, cmap='tab10')
        axes[0, 1].set_title('Raw Predicted Classes', fontweight='bold', fontsize=14)
        axes[0, 1].axis('off')
        
        # Show probability maps for each class
        for i, class_name in enumerate(class_names):
            if i < 2:  # Show first 2 classes
                if i < probs_np.shape[0]:
                    im = axes[0, 2 + i].imshow(probs_np[i], cmap='hot', vmin=0, vmax=1)
                    axes[0, 2 + i].set_title(f'{class_name} Raw Prob', fontweight='bold', fontsize=14)
                    axes[0, 2 + i].axis('off')
                    plt.colorbar(im, ax=axes[0, 2 + i], fraction=0.046, pad=0.04)
                else:
                    axes[0, 2 + i].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                      transform=axes[0, 2 + i].transAxes)
                    axes[0, 2 + i].set_title(f'{class_name} Raw Prob', fontweight='bold', fontsize=14)
                    axes[0, 2 + i].axis('off')
        
        # Row 2: Class-specific analysis
        for i, class_name in enumerate(class_names):
            if i < 4:
                if i < probs_np.shape[0]:
                    # Create binary mask for this class
                    class_mask = (pred_classes == i).astype(np.uint8)
                    
                    # Show probability map
                    im = axes[1, i].imshow(probs_np[i], cmap='hot', vmin=0, vmax=1)
                    axes[1, i].set_title(f'{class_name} Probability Map', fontweight='bold', fontsize=12)
                    axes[1, i].axis('off')
                    plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
                else:
                    axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                                  transform=axes[1, i].transAxes)
                    axes[1, i].set_title(f'{class_name} Probability Map', fontweight='bold', fontsize=12)
                    axes[1, i].axis('off')
        
        # Row 3: Statistics and analysis
        # Class distribution
        unique_classes, class_counts = np.unique(pred_classes, return_counts=True)
        axes[2, 0].bar(unique_classes, class_counts)
        axes[2, 0].set_title('Class Distribution', fontweight='bold', fontsize=12)
        axes[2, 0].set_xlabel('Class Index')
        axes[2, 0].set_ylabel('Pixel Count')
        
        # Probability statistics
        prob_means = [np.mean(probs_np[i]) for i in range(min(4, probs_np.shape[0]))]
        prob_maxs = [np.max(probs_np[i]) for i in range(min(4, probs_np.shape[0]))]
        
        x = np.arange(len(prob_means))
        width = 0.35
        
        axes[2, 1].bar(x - width/2, prob_means, width, label='Mean Prob', alpha=0.8)
        axes[2, 1].bar(x + width/2, prob_maxs, width, label='Max Prob', alpha=0.8)
        axes[2, 1].set_title('Probability Statistics', fontweight='bold', fontsize=12)
        axes[2, 1].set_xlabel('Class')
        axes[2, 1].set_ylabel('Probability')
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels(class_names[:len(prob_means)])
        axes[2, 1].legend()
        
        # Confidence analysis
        max_probs = np.max(probs_np, axis=0)
        confidence_hist = axes[2, 2].hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
        axes[2, 2].set_title('Confidence Distribution', fontweight='bold', fontsize=12)
        axes[2, 2].set_xlabel('Max Probability')
        axes[2, 2].set_ylabel('Pixel Count')
        
        # Summary statistics
        summary_text = f"Model Output Summary\n"
        summary_text += f"Image Shape: {original_image.shape}\n"
        summary_text += f"Predicted Classes: {len(unique_classes)}\n"
        summary_text += f"Class Range: {pred_classes.min()}-{pred_classes.max()}\n"
        summary_text += f"Probability Range: {probs_np.min():.4f}-{probs_np.max():.4f}\n"
        summary_text += f"Avg Confidence: {np.mean(max_probs):.4f}\n"
        summary_text += f"High Confidence (>0.8): {np.sum(max_probs > 0.8)} pixels\n"
        summary_text += f"Low Confidence (<0.5): {np.sum(max_probs < 0.5)} pixels"
        
        axes[2, 3].text(0.1, 0.9, summary_text, transform=axes[2, 3].transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[2, 3].set_title('Summary Statistics', fontweight='bold', fontsize=12)
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        # Save detailed comparison
        detailed_path = "detailed_model_output_comparison.png"
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Detailed comparison saved to: {detailed_path}")
        
    except Exception as e:
        print(f"❌ Error creating detailed comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🚀 Starting Model Output Debug")
    print("=" * 60)
    
    success = debug_model_output()
    
    if success:
        print("\n🎉 Model output debugging completed!")
        print("✅ Check the generated images to see raw model predictions")
    else:
        print("\n💥 Model output debugging failed!")
        print("⚠️ Check the error messages above for details")