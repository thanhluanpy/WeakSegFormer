#!/usr/bin/env python3
"""
Compare results between synthetic image and real MRI
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

def compare_synthetic_vs_real():
    """Compare synthetic vs real MRI results"""
    print("🔍 Comparing Synthetic vs Real MRI Results")
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
        
        # Test images
        synthetic_path = "test_mri.jpg"
        real_mri_path = r"F:\MRI-Result\BraTS-GLI-00002-000_78.jpg"
        
        # Check if files exist
        if not os.path.exists(synthetic_path):
            print(f"❌ Synthetic image not found: {synthetic_path}")
            return False
        
        if not os.path.exists(real_mri_path):
            print(f"❌ Real MRI not found: {real_mri_path}")
            return False
        
        print(f"✅ Found synthetic image: {synthetic_path}")
        print(f"✅ Found real MRI: {real_mri_path}")
        
        # Get model
        from app import model, device
        model.eval()
        
        # Test both images
        results = {}
        
        for image_name, image_path in [("Synthetic", synthetic_path), ("Real MRI", real_mri_path)]:
            print(f"\n🔄 Testing {image_name}...")
            
            # Load image
            original_image = np.array(Image.open(image_path))
            print(f"   - Original shape: {original_image.shape}")
            
            # Preprocess
            image_tensor, processed_image = preprocess_image(image_path)
            
            # Run inference
            with torch.no_grad():
                output = model(image_tensor.to(device))
                
                if isinstance(output, (list, tuple)):
                    main_output = output[0]
                else:
                    main_output = output
            
            # Apply softmax
            probs = torch.softmax(main_output, dim=1)
            probs_np = probs.cpu().numpy().squeeze()
            
            # Get predicted classes
            pred_classes = np.argmax(probs_np, axis=0)
            
            # Analyze classes
            class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
            class_stats = {}
            
            for i, class_name in enumerate(class_names):
                if i < probs_np.shape[0]:
                    class_pixels = np.sum(pred_classes == i)
                    total_pixels = pred_classes.size
                    percentage = (class_pixels / total_pixels) * 100
                    avg_prob = np.mean(probs_np[i])
                    max_prob = np.max(probs_np[i])
                    
                    class_stats[class_name] = {
                        'pixels': class_pixels,
                        'percentage': percentage,
                        'avg_prob': avg_prob,
                        'max_prob': max_prob
                    }
            
            results[image_name] = {
                'original_image': original_image,
                'processed_image': processed_image,
                'pred_classes': pred_classes,
                'probs': probs_np,
                'class_stats': class_stats
            }
            
            print(f"   - Unique classes: {np.unique(pred_classes)}")
            for class_name, stats in class_stats.items():
                print(f"   - {class_name}: {stats['pixels']:,} pixels ({stats['percentage']:.1f}%)")
        
        # Create comparison visualization
        print("\n🎨 Creating comparison visualization...")
        create_comparison_visualization(results)
        
        # Create statistics comparison
        print("\n📊 Creating statistics comparison...")
        create_statistics_comparison(results)
        
        print(f"\n✅ Comparison completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_visualization(results):
    """Create side-by-side comparison visualization"""
    try:
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('Synthetic vs Real MRI Comparison', fontsize=18, fontweight='bold')
        
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        colors = ['black', 'darkred', 'green', 'blue']
        
        # Row 1: Original images
        axes[0, 0].imshow(results['Synthetic']['original_image'])
        axes[0, 0].set_title('Synthetic - Original', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['Real MRI']['original_image'])
        axes[0, 1].set_title('Real MRI - Original', fontweight='bold', fontsize=14)
        axes[0, 1].axis('off')
        
        # Row 2: Processed images
        axes[1, 0].imshow(results['Synthetic']['processed_image'])
        axes[1, 0].set_title('Synthetic - Processed', fontweight='bold', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['Real MRI']['processed_image'])
        axes[1, 1].set_title('Real MRI - Processed', fontweight='bold', fontsize=14)
        axes[1, 1].axis('off')
        
        # Row 3: Predicted classes
        im1 = axes[2, 0].imshow(results['Synthetic']['pred_classes'], cmap='tab10')
        axes[2, 0].set_title('Synthetic - Predicted Classes', fontweight='bold', fontsize=14)
        axes[2, 0].axis('off')
        plt.colorbar(im1, ax=axes[2, 0], fraction=0.046, pad=0.04)
        
        im2 = axes[2, 1].imshow(results['Real MRI']['pred_classes'], cmap='tab10')
        axes[2, 1].set_title('Real MRI - Predicted Classes', fontweight='bold', fontsize=14)
        axes[2, 1].axis('off')
        plt.colorbar(im2, ax=axes[2, 1], fraction=0.046, pad=0.04)
        
        # Row 4: Class distribution comparison
        synthetic_stats = results['Synthetic']['class_stats']
        real_stats = results['Real MRI']['class_stats']
        
        # Synthetic distribution
        synthetic_percentages = [synthetic_stats[class_name]['percentage'] for class_name in class_names]
        bars1 = axes[3, 0].bar(class_names, synthetic_percentages, color=colors)
        axes[3, 0].set_title('Synthetic - Class Distribution', fontweight='bold', fontsize=14)
        axes[3, 0].set_ylabel('Percentage (%)')
        axes[3, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, pct in zip(bars1, synthetic_percentages):
            height = bar.get_height()
            axes[3, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Real MRI distribution
        real_percentages = [real_stats[class_name]['percentage'] for class_name in class_names]
        bars2 = axes[3, 1].bar(class_names, real_percentages, color=colors)
        axes[3, 1].set_title('Real MRI - Class Distribution', fontweight='bold', fontsize=14)
        axes[3, 1].set_ylabel('Percentage (%)')
        axes[3, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, pct in zip(bars2, real_percentages):
            height = bar.get_height()
            axes[3, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Hide unused subplots
        for i in range(2, 4):
            for j in range(2, 4):
                axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        comparison_path = "synthetic_vs_real_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison visualization saved to: {comparison_path}")
        
    except Exception as e:
        print(f"❌ Error creating comparison visualization: {e}")

def create_statistics_comparison(results):
    """Create detailed statistics comparison"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Statistics Comparison', fontsize=16, fontweight='bold')
        
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        synthetic_stats = results['Synthetic']['class_stats']
        real_stats = results['Real MRI']['class_stats']
        
        # 1. Percentage comparison
        synthetic_percentages = [synthetic_stats[class_name]['percentage'] for class_name in class_names]
        real_percentages = [real_stats[class_name]['percentage'] for class_name in class_names]
        
        x = np.arange(len(class_names))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, synthetic_percentages, width, label='Synthetic', alpha=0.8)
        bars2 = axes[0, 0].bar(x + width/2, real_percentages, width, label='Real MRI', alpha=0.8)
        
        axes[0, 0].set_title('Class Distribution Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Percentage (%)')
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names)
        axes[0, 0].legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. Average probability comparison
        synthetic_avg_probs = [synthetic_stats[class_name]['avg_prob'] for class_name in class_names]
        real_avg_probs = [real_stats[class_name]['avg_prob'] for class_name in class_names]
        
        bars3 = axes[0, 1].bar(x - width/2, synthetic_avg_probs, width, label='Synthetic', alpha=0.8)
        bars4 = axes[0, 1].bar(x + width/2, real_avg_probs, width, label='Real MRI', alpha=0.8)
        
        axes[0, 1].set_title('Average Probability Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('Average Probability')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_names)
        axes[0, 1].legend()
        
        # 3. Max probability comparison
        synthetic_max_probs = [synthetic_stats[class_name]['max_prob'] for class_name in class_names]
        real_max_probs = [real_stats[class_name]['max_prob'] for class_name in class_names]
        
        bars5 = axes[1, 0].bar(x - width/2, synthetic_max_probs, width, label='Synthetic', alpha=0.8)
        bars6 = axes[1, 0].bar(x + width/2, real_max_probs, width, label='Real MRI', alpha=0.8)
        
        axes[1, 0].set_title('Max Probability Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Max Probability')
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names)
        axes[1, 0].legend()
        
        # 4. Summary table
        axes[1, 1].axis('off')
        
        # Create summary table
        summary_data = []
        for class_name in class_names:
            synthetic_pct = synthetic_stats[class_name]['percentage']
            real_pct = real_stats[class_name]['percentage']
            difference = real_pct - synthetic_pct
            
            summary_data.append([
                class_name,
                f"{synthetic_pct:.1f}%",
                f"{real_pct:.1f}%",
                f"{difference:+.1f}%"
            ])
        
        table = axes[1, 1].table(cellText=summary_data,
                                colLabels=['Class', 'Synthetic', 'Real MRI', 'Difference'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        axes[1, 1].set_title('Summary Comparison', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save statistics comparison
        stats_path = "synthetic_vs_real_statistics.png"
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Statistics comparison saved to: {stats_path}")
        
    except Exception as e:
        print(f"❌ Error creating statistics comparison: {e}")

if __name__ == '__main__':
    print("🚀 Starting Synthetic vs Real MRI Comparison")
    print("=" * 60)
    
    success = compare_synthetic_vs_real()
    
    if success:
        print("\n🎉 Comparison completed successfully!")
        print("✅ Check the generated images to see detailed comparison")
    else:
        print("\n💥 Comparison failed!")
        print("⚠️ Check the error messages above for details")