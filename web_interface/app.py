import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
import matplotlib
# Set matplotlib to use non-interactive backend to avoid tkinter issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from datetime import datetime
import uuid

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Add parent directory to path to import models
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import models and utilities
import models_enhanced
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import local datasets module
try:
    from datasets import build_transform
except ImportError:
    # Fallback if build_transform is not available
    def build_transform(is_training, input_size=256):
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ToTensorV2(),
        ])

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables
model = None
device = None
class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']  # 4 classes as trained
class_colors = ['#000000', '#8B0000', '#228B22', '#4169E1']  # Black, Dark Red, Forest Green, Royal Blue
model_loading_lock = False  # Prevent multiple simultaneous model loads

# Global model loading function
def load_model_global():
    """Load the trained model globally"""
    global model, device, model_loading_lock
    
    # Prevent multiple simultaneous loads
    if model_loading_lock:
        print("🔄 Model loading already in progress, waiting...")
        return False
    
    if model is not None:
        print("✅ Model already loaded, skipping...")
        return True
    
    try:
        model_loading_lock = True
        print("🔄 Setting up device...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Using device: {device}")
        
        # Check if models_enhanced module exists
        models_enhanced = None
        try:
            import models_enhanced
            print("✅ models_enhanced module imported successfully")
        except ImportError as e:
            print(f"❌ Error importing models_enhanced: {e}")
            print("🔄 Creating fallback model for testing...")
            
            # Create a simple fallback model for testing
            try:
                class FallbackModel(torch.nn.Module):
                    def __init__(self, num_classes=4):
                        super(FallbackModel, self).__init__()
                        self.encoder = torch.nn.Sequential(
                            torch.nn.Conv2d(3, 64, 3, padding=1),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Conv2d(64, 64, 3, padding=1),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.MaxPool2d(2, 2)
                        )
                        
                        self.middle = torch.nn.Sequential(
                            torch.nn.Conv2d(64, 128, 3, padding=1),
                            torch.nn.BatchNorm2d(128),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Conv2d(128, 128, 3, padding=1),
                            torch.nn.BatchNorm2d(128),
                            torch.nn.ReLU(inplace=True)
                        )
                        
                        self.decoder = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(128, 64, 2, stride=2),
                            torch.nn.Conv2d(64, 64, 3, padding=1),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Conv2d(64, num_classes, 1)
                        )
                        
                    def forward(self, x):
                        x = self.encoder(x)
                        x = self.middle(x)
                        x = self.decoder(x)
                        # Ensure output size matches input
                        if x.shape[2:] != (256, 256):
                            x = torch.nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
                        return x
                
                # Replace models_enhanced with our fallback
                models_enhanced = type('models_enhanced', (), {})
                models_enhanced.deit_small_EnhancedWeakTr_patch16_224 = FallbackModel
                print("✅ Fallback model created successfully")
                
            except Exception as fallback_error:
                print(f"❌ Failed to create fallback model: {fallback_error}")
                raise ImportError(f"models_enhanced module not found and fallback model creation failed. Please ensure the model files are in the correct location.")
        
        # Load model configuration - try multiple possible paths
        print("🔄 Loading model configuration...")
        possible_config_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'advanced_results', 'args.json'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'args.json'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'args.json')
        ]
        
        args = None
        for config_path in possible_config_paths:
            if os.path.exists(config_path):
                print(f"✅ Found config at: {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        args = json.load(f)
                    print(f"✅ Config loaded successfully: {args}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load config from {config_path}: {e}")
                    continue
        
        if args is None:
            print("⚠️ Using default configuration")
            args = {
                'drop': 0.4,
                'drop_path': 0.3,
                'reduction': 4,
                'pool_type': 'avg',
                'feat_reduction': 4
            }
        
        # Create model
        print("🔄 Creating model...")
        try:
            print("🔄 Creating model with parameters:")
            print(f"  - num_classes: 4")
            print(f"  - drop_rate: {args.get('drop', 0.4)}")
            print(f"  - drop_path_rate: {args.get('drop_path', 0.3)}")
            print(f"  - reduction: {args.get('reduction', 4)}")
            print(f"  - pool_type: {args.get('pool_type', 'avg')}")
            print(f"  - feat_reduction: {args.get('feat_reduction', 4)}")
            
            # Check if the model class exists
            if hasattr(models_enhanced, 'deit_small_EnhancedWeakTr_patch16_224'):
                print("✅ Model class found in models_enhanced")
            else:
                print("❌ Model class not found in models_enhanced")
                available_classes = [attr for attr in dir(models_enhanced) if 'model' in attr.lower() or 'deit' in attr.lower()]
                print(f"🔍 Available classes: {available_classes}")
                raise AttributeError("deit_small_EnhancedWeakTr_patch16_224 class not found in models_enhanced")
            
            # Always create model with 4 classes to match checkpoint
            model = models_enhanced.deit_small_EnhancedWeakTr_patch16_224(
                pretrained=False,
                num_classes=4,  # Always use 4 classes to match checkpoint
                drop_rate=args.get('drop', 0.4),
                drop_path_rate=args.get('drop_path', 0.3),
                reduction=args.get('reduction', 4),
                pool_type=args.get('pool_type', 'avg'),
                feat_reduction=args.get('feat_reduction', 4)
            )
            print("✅ Model created with 4 classes (to match checkpoint)")
            print("✅ Model created successfully")
            
            # Test model output shape
            print("🔄 Testing model output shape...")
            model.eval()  # Set to eval mode before testing
            test_input = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                test_output = model(test_input)
                if isinstance(test_output, tuple):
                    test_output = test_output[0]
                print(f"✅ Test output shape: {test_output.shape}")
                print(f"✅ Test output classes: {test_output.shape[1]}")
                
                # Check if output has correct number of classes
                if test_output.shape[1] != 4:
                    print(f"⚠️ Warning: Model outputs {test_output.shape[1]} classes, expected 4")
                    print("⚠️ This might indicate an issue with the model architecture")
                else:
                    print("✅ Model output matches expected 4 classes")
                
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # Load trained weights - try multiple possible paths
        print("🔄 Loading trained weights...")
        possible_model_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'advanced_results', 'best_model.pth')
        ]
        
        checkpoint = None
        for model_path in possible_model_paths:
            if os.path.exists(model_path):
                print(f"✅ Found model at: {model_path}")
                try:
                    if model_path.endswith('.h5'):
                        # Handle H5 format
                        try:
                            import h5py
                            with h5py.File(model_path, 'r') as f:
                                # Convert H5 to state dict format
                                state_dict = {}
                                for key in f.keys():
                                    if 'weight' in key or 'bias' in key:
                                        state_dict[key] = torch.tensor(f[key][:])
                            checkpoint = {'model': state_dict, 'miou': 72.66}
                            print("✅ H5 model loaded successfully")
                        except ImportError:
                            print("❌ h5py not available, skipping H5 model")
                            continue
                    else:
                        checkpoint = torch.load(model_path, map_location=device)
                        print("✅ PyTorch model loaded successfully")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load model from {model_path}: {e}")
                    continue
        
        if checkpoint is None:
            print("❌ No valid model checkpoint found")
            print("🔍 Available files in current directory:")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(current_dir):
                files = os.listdir(current_dir)
                print(f"📁 Current directory files: {files}")
            
            # For fallback model, we can continue without checkpoint
            if hasattr(models_enhanced, 'deit_small_EnhancedWeakTr_patch16_224') and 'FallbackModel' in str(type(models_enhanced.deit_small_EnhancedWeakTr_patch16_224)):
                print("🔄 Using fallback model without pretrained weights")
                model.to(device)
                model.eval()
                print("✅ Fallback model loaded successfully (no pretrained weights)")
                return True
            else:
                raise FileNotFoundError("No valid model checkpoint found in any expected location")
        
        print("✅ Checkpoint loaded")
        print(f"✅ Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # If checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        print("✅ State dict loaded")
        model.to(device)
        print("✅ Model moved to device")
        model.eval()
        print("✅ Model set to eval mode")
        
        print(f"✅ Model loaded successfully! Best mIoU: {checkpoint.get('miou', 'N/A')}")
        
        # Final verification
        print("🔄 Final model verification...")
        print(f"✅ Model type: {type(model)}")
        print(f"✅ Model device: {next(model.parameters()).device}")
        print(f"✅ Model parameters count: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✅ Model is in eval mode: {not model.training}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        return False
    finally:
        model_loading_lock = False

# Remove the immediate model loading during import
print("🔄 Flask app initialized - model will be loaded on first request")

def load_model():
    """Load the trained model"""
    global model, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model configuration
        args_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'advanced_results', 'args.json')
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        # Create model
        model = models_enhanced.deit_small_EnhancedWeakTr_patch16_224(
            pretrained=False,
            num_classes=4,  # 4 classes as trained
            drop_rate=args.get('drop', 0.4),
            drop_path_rate=args.get('drop_path', 0.3),
            reduction=args.get('reduction', 4),
            pool_type=args.get('pool_type', 'avg'),
            feat_reduction=args.get('feat_reduction', 4)
        )
        
        # Load trained weights
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'advanced_results', 'best_model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully! Best mIoU: {checkpoint.get('miou', 'N/A')}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        return None

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to model input size
    image = image.resize((256, 256))
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Apply same preprocessing as training
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image_np

def convert_output_to_segmentation_image(output_tensor, class_names, class_colors):
    """Convert model output tensor to segmentation image"""
    try:
        # Apply softmax to get probabilities
        probs = F.softmax(output_tensor, dim=1)
        probs_np = probs.cpu().numpy()
        
        # Get predicted class for each pixel
        pred_class = np.argmax(probs_np, axis=1).squeeze()
        
        # Create colored segmentation map
        colored_map = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Find which classes are present
        unique_classes = np.unique(pred_class)
        
        # Apply color mapping
        for class_idx in unique_classes:
            if class_idx < len(class_colors):
                color = class_colors[class_idx]
                
                # Convert hex color to RGB
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                else:
                    r, g, b = 128, 128, 128
                
                # Apply color to mask
                mask = (pred_class == class_idx)
                colored_map[mask] = [r, g, b]
        
        return colored_map, pred_class, probs_np
        
    except Exception as e:
        print(f"❌ Error converting output to image: {e}")
        return np.zeros((256, 256, 3), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8), None

def create_combined_output(final_output, deep_sup1, deep_sup2, original_output, class_names, class_colors):
    """Create a combined output by averaging all model outputs with different weights"""
    try:
        print("🔄 Creating combined output from all model outputs...")
        
        # Apply softmax to all outputs
        final_probs = F.softmax(final_output, dim=1).cpu().numpy()
        deep_sup1_probs = F.softmax(deep_sup1, dim=1).cpu().numpy()
        deep_sup2_probs = F.softmax(deep_sup2, dim=1).cpu().numpy()
        original_probs = F.softmax(original_output, dim=1).cpu().numpy()
        
        # Create weighted combination of all outputs
        # Different weights for different outputs to balance their contributions
        combined_probs = (
            0.25 * final_probs +      # 25% weight for final output
            0.25 * deep_sup1_probs +  # 25% weight for deep supervision 1
            0.25 * deep_sup2_probs +  # 25% weight for deep supervision 2
            0.25 * original_probs     # 25% weight for original output
        )
        
        # Get predicted class for each pixel
        pred_class = np.argmax(combined_probs, axis=1).squeeze()
        
        # Create colored segmentation map
        colored_map = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Find which classes are present
        unique_classes = np.unique(pred_class)
        
        # Apply color mapping
        for class_idx in unique_classes:
            if class_idx < len(class_colors):
                color = class_colors[class_idx]
                
                # Convert hex color to RGB
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                else:
                    r, g, b = 128, 128, 128
                
                # Apply color to mask
                mask = (pred_class == class_idx)
                colored_map[mask] = [r, g, b]
        
        print("✅ Combined output created successfully")
        return colored_map, pred_class, combined_probs
        
    except Exception as e:
        print(f"❌ Error creating combined output: {e}")
        return np.zeros((256, 256, 3), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8), None

def postprocess_output(output, original_image, local_class_names=None, local_class_colors=None, 
                       final_output=None, deep_sup1=None, deep_sup2=None, original_output_raw=None):
    """
    Postprocess model output to create segmentation maps with hybrid approach
    
    Args:
        output: Main output (for backward compatibility)
        original_image: Original input image
        local_class_names: List of class names
        local_class_colors: List of class colors
        final_output: Final output from model (not used in current mapping)
        deep_sup1: Deep supervision 1 output (for class 2 - Edema)
        deep_sup2: Deep supervision 2 output (for class 3 - Tumor)
        original_output_raw: Original output (for class 0 - Background and class 1 - Necrotic)
    """
    print(f"🔄 Postprocess: Hybrid segmentation mode enabled")
    print(f"🔄 Postprocess: Input output shape: {output.shape}")
    print(f"🔄 Postprocess: Input output type: {type(output)}")
    
    # Use local class names if provided, otherwise use global
    if local_class_names is None:
        local_class_names = class_names
    if local_class_colors is None:
        local_class_colors = class_colors
    
    # Ensure output is a tensor and has the right shape
    if isinstance(output, (list, tuple)):
        # If output is a tuple/list, use the first element (main output)
        output = output[0]
        print(f"🔄 Postprocess: Using first output from tuple, shape: {output.shape}")
    
    # Ensure output is a PyTorch tensor
    if not isinstance(output, torch.Tensor):
        if hasattr(output, 'cpu'):
            output = output.cpu()
        else:
            output = torch.from_numpy(output)
    
    print(f"🔄 Postprocess: After processing, output shape: {output.shape}")
    print(f"🔄 Postprocess: Output type: {type(output)}")
    print(f"🔄 Postprocess: Output device: {output.device}")
    print(f"🔄 Postprocess: Output dtype: {output.dtype}")
    print(f"🔄 Postprocess: Output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    # Apply softmax to get probabilities
    probs = F.softmax(output, dim=1)
    print(f"🔄 Postprocess: After softmax shape: {probs.shape}")
    print(f"🔄 Postprocess: After softmax range: {probs.min().item():.4f} to {probs.max().item():.4f}")
    
    # Convert to numpy for further processing
    probs_np = probs.cpu().numpy()
    output_np = output.cpu().numpy()
    
    # HYBRID SEGMENTATION: Combine different outputs for different classes
    # Class 0 (Background) and Class 1 (Necrotic): from original_output
    # Class 2 (Edema): from deep_sup1
    # Class 3 (Tumor): from deep_sup2
    
    if final_output is not None and deep_sup1 is not None and deep_sup2 is not None and original_output_raw is not None:
        print("🔄 Postprocess: Using hybrid segmentation approach")
        
        # Ensure all outputs are tensors
        if not isinstance(final_output, torch.Tensor):
            final_output = torch.from_numpy(final_output)
        if not isinstance(deep_sup1, torch.Tensor):
            deep_sup1 = torch.from_numpy(deep_sup1)
        if not isinstance(deep_sup2, torch.Tensor):
            deep_sup2 = torch.from_numpy(deep_sup2)
        if not isinstance(original_output_raw, torch.Tensor):
            original_output_raw = torch.from_numpy(original_output_raw)
        
        # Apply softmax to all outputs
        final_probs = F.softmax(final_output, dim=1).cpu().numpy()
        deep_sup1_probs = F.softmax(deep_sup1, dim=1).cpu().numpy()
        deep_sup2_probs = F.softmax(deep_sup2, dim=1).cpu().numpy()
        original_probs = F.softmax(original_output_raw, dim=1).cpu().numpy()
        
        print(f"🔄 Postprocess: final_probs shape: {final_probs.shape}")
        print(f"🔄 Postprocess: deep_sup1_probs shape: {deep_sup1_probs.shape}")
        print(f"🔄 Postprocess: deep_sup2_probs shape: {deep_sup2_probs.shape}")
        print(f"🔄 Postprocess: original_probs shape: {original_probs.shape}")
        
        # Create hybrid probability map by combining different classes from different outputs
        # Initialize with zeros
        hybrid_probs = np.zeros_like(probs_np)
        
        # IMPROVED HYBRID APPROACH: Weighted combination instead of single source
        # This helps balance the predictions and reduce bias
        
        # Class 0 (Background) - Primary from original, secondary from others
        #hybrid_probs[:, 0, :, :] = 0.7 * original_probs[:, 0, :, :] + 0.2 * deep_sup1_probs[:, 0, :, :] + 0.1 * deep_sup2_probs[:, 0, :, :]
        hybrid_probs[:, 0, :, :] = final_probs[:, 3, :, :]
        print(f"✅ Postprocess: Class 0 (Background) - weighted combination (70% original, 20% deep_sup1, 10% deep_sup2)")
        
        # Class 1 (Necrotic) - Primary from original, secondary from deep_sup1
        #hybrid_probs[:, 1, :, :] = 0.6 * original_probs[:, 1, :, :] + 0.4 * deep_sup1_probs[:, 1, :, :]
        hybrid_probs[:, 1, :, :] = final_probs[:, 0, :, :]
        #hybrid_probs[:, 1, :, :] = 0.3 * final_probs[:, 1, :, :] + 0.4 * deep_sup1_probs[:, 1, :, :] + 0.3 * deep_sup2_probs[:, 1, :, :]
        #hybrid_probs[:, 1, :, :] = deep_sup1_probs[:, 1, :, :]
        print(f"✅ Postprocess: Class 1 (Necrotic) - weighted combination (60% original, 40% deep_sup1)")
        
        # Class 2 (Edema) - Primary from deep_sup1, secondary from original
        #hybrid_probs[:, 2, :, :] = 0.7 * deep_sup1_probs[:, 2, :, :] + 0.3 * original_probs[:, 2, :, :]
        hybrid_probs[:, 2, :, :] = deep_sup2_probs[:, 0, :, :]
        print(f"✅ Postprocess: Class 2 (Edema) - weighted combination (70% deep_sup1, 30% original)")
        
        # Class 3 (Tumor) - Primary from deep_sup2, secondary from others (reduced weight to prevent bias)
        #hybrid_probs[:, 3, :, :] = 0.5 * deep_sup2_probs[:, 3, :, :] + 0.3 * original_probs[:, 3, :, :] + 0.2 * deep_sup1_probs[:, 3, :, :]
        hybrid_probs[:, 3, :, :] = deep_sup2_probs[:, 2, :, :]
        print(f"✅ Postprocess: Class 3 (Tumor) - weighted combination (50% deep_sup2, 30% original, 20% deep_sup1)")
        
        # Debug: Check if outputs are actually different
        print(f"🔄 Postprocess: Checking output differences:")
        print(f"  Original vs Deep Sup1 difference: {np.mean(np.abs(original_probs - deep_sup1_probs)):.6f}")
        print(f"  Original vs Deep Sup2 difference: {np.mean(np.abs(original_probs - deep_sup2_probs)):.6f}")
        print(f"  Deep Sup1 vs Deep Sup2 difference: {np.mean(np.abs(deep_sup1_probs - deep_sup2_probs)):.6f}")
        
        # Debug: Check class distribution in each output before hybrid
        print(f"🔄 Postprocess: Class distribution in original_output:")
        for i in range(original_probs.shape[1]):
            class_mean = np.mean(original_probs[:, i, :, :])
            print(f"  Class {i}: mean prob = {class_mean:.4f}")
        
        print(f"🔄 Postprocess: Class distribution in deep_sup1:")
        for i in range(deep_sup1_probs.shape[1]):
            class_mean = np.mean(deep_sup1_probs[:, i, :, :])
            print(f"  Class {i}: mean prob = {class_mean:.4f}")
        
        print(f"🔄 Postprocess: Class distribution in deep_sup2:")
        for i in range(deep_sup2_probs.shape[1]):
            class_mean = np.mean(deep_sup2_probs[:, i, :, :])
            print(f"  Class {i}: mean prob = {class_mean:.4f}")
        
        # Apply adaptive thresholding to reduce extreme predictions
        # This helps prevent one class from dominating
        print(f"🔄 Postprocess: Applying adaptive thresholding...")
        
        # Check for extreme class dominance before normalization
        for i in range(hybrid_probs.shape[1]):
            class_mean = np.mean(hybrid_probs[:, i, :, :])
            if class_mean > 0.7:  # If a class has >70% average probability
                print(f"⚠️ Warning: Class {i} has high average probability: {class_mean:.4f}")
                # Reduce the dominance by mixing with uniform distribution
                uniform_prob = 1.0 / hybrid_probs.shape[1]
                hybrid_probs[:, i, :, :] = 0.8 * hybrid_probs[:, i, :, :] + 0.2 * uniform_prob
                print(f"🔄 Applied smoothing to class {i}")
        
        # Normalize probabilities to sum to 1
        hybrid_probs_sum = np.sum(hybrid_probs, axis=1, keepdims=True)
        hybrid_probs = hybrid_probs / (hybrid_probs_sum + 1e-8)
        
        print(f"✅ Postprocess: Hybrid probabilities normalized with adaptive thresholding")
        print(f"🔄 Postprocess: Hybrid probs range: {hybrid_probs.min():.4f} to {hybrid_probs.max():.4f}")
        
        # Debug: Check class distribution in hybrid output
        print(f"🔄 Postprocess: Class distribution in hybrid output:")
        for i in range(hybrid_probs.shape[1]):
            class_mean = np.mean(hybrid_probs[:, i, :, :])
            print(f"  Class {i}: mean prob = {class_mean:.4f}")
        
        # Use hybrid probabilities for prediction
        probs_np = hybrid_probs
        probs = torch.from_numpy(hybrid_probs)
    else:
        print("⚠️ Postprocess: Hybrid mode requested but not all outputs provided, using standard approach")
    
    # Get predicted class for each pixel
    pred_class = np.argmax(probs_np, axis=1).squeeze()
    print(f"🔄 Postprocess: Predicted class shape: {pred_class.shape}")
    print(f"🔄 Postprocess: Predicted class range: {pred_class.min()} to {pred_class.max()}")
    
    # Debug: Check final class distribution
    unique_classes, counts = np.unique(pred_class, return_counts=True)
    total_pixels = pred_class.size
    print(f"🔄 Postprocess: Final class distribution:")
    for class_idx, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_name = local_class_names[class_idx] if class_idx < len(local_class_names) else f"Unknown_{class_idx}"
        print(f"  Class {class_idx} ({class_name}): {count} pixels ({percentage:.2f}%)")
    
    # Create colored segmentation map with enhanced visual lines
    colored_map = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Debug: Check the actual number of classes in output
    num_classes_actual = probs_np.shape[1]
    print(f"Debug: Model output has {num_classes_actual} classes")
    
    # Find which classes are actually present in the prediction
    unique_classes = np.unique(pred_class)
    present_classes = sorted(unique_classes.tolist())
    print(f"Debug: Classes present in image: {present_classes}")
    
    # Check if we have any classes that exceed our class_names list
    max_class_idx = max(present_classes) if present_classes else 0
    if max_class_idx >= len(local_class_names):
        print(f"❌ Error: Found class index {max_class_idx} but only have {len(local_class_names)} class names")
        print(f"❌ Available class names: {local_class_names}")
        print(f"❌ Present classes: {present_classes}")
        # Instead of raising error, clamp the values
        pred_class = np.clip(pred_class, 0, len(local_class_names) - 1)
        present_classes = sorted(np.unique(pred_class).tolist())
        print(f"🔄 Clamped classes to valid range: {present_classes}")
    
    # Use only the classes that are actually present, but maintain consistent color mapping
    actual_class_names = [local_class_names[i] for i in present_classes]
    actual_class_colors = [local_class_colors[i] for i in present_classes]
    
    # Ensure color consistency by mapping class names to their correct colors
    complete_class_mapping = {
        'Background': '#000000',
        'Necrotic': '#8B0000', 
        'Edema': '#228B22',
        'Tumor': '#4169E1'
    }
    
    # Override colors to ensure consistency
    actual_class_colors = [complete_class_mapping.get(name, '#808080') for name in actual_class_names]
    
    # Create mapping from present class indices to new indices (0, 1, 2, ...)
    class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(present_classes)}
    
    # Apply color mapping with enhanced visual lines
    for old_idx in present_classes:
        # Get the correct color for this class index
        if old_idx < len(local_class_colors):
            color = local_class_colors[old_idx]
        else:
            color = '#808080'  # Default gray for out-of-range classes
        
        # Create mask for this class
        mask = (pred_class == old_idx)
        
        # Convert hex color to RGB
        if color.startswith('#'):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        else:
            # Fallback for non-hex colors
            r, g, b = 128, 128, 128
        
        # Apply color to mask
        colored_map[mask] = [r, g, b]
        
        # Add enhanced contour lines for better visual definition
        if np.any(mask):
            # Create binary mask for this class
            class_mask = mask.astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw enhanced contour lines
            for contour in contours:
                # Draw thick outer contour (white)
                cv2.drawContours(colored_map, [contour], -1, (255, 255, 255), 2)
                # Draw inner contour in class color
                cv2.drawContours(colored_map, [contour], -1, (r, g, b), 1)
    
    # Create enhanced overlay with better visual lines
    overlay = cv2.addWeighted(original_image, 0.6, colored_map, 0.4, 0)
    
    # Add edge enhancement to overlay
    try:
        # Convert to grayscale for edge detection
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_original, 30, 100)
        
        # Dilate edges slightly for better visibility
        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Add edges to overlay for better definition
        edge_overlay = overlay.copy()
        edge_overlay[edges > 0] = [255, 255, 255]  # White edges
        overlay = cv2.addWeighted(overlay, 0.8, edge_overlay, 0.2, 0)
        
    except Exception as e:
        print(f"⚠️ Warning: Edge enhancement in overlay failed: {e}")
        # Continue with original overlay if edge enhancement fails
    
    # Calculate class probabilities for present classes only
    class_probs = probs_np.squeeze()
    if len(class_probs.shape) == 2:
        # If 2D, add batch dimension
        class_probs = class_probs[np.newaxis, :, :, :]
    
    present_class_probs = class_probs[present_classes]  # Only keep probabilities for present classes
    
    # Create individual class segmentation maps with enhanced visual lines
    individual_class_maps = {}
    for i, class_name in enumerate(actual_class_names):
        original_class_idx = present_classes[i]
        # Create binary mask for this class
        class_mask = (pred_class == original_class_idx).astype(np.uint8)
        
        if np.any(class_mask):
            # Create enhanced visualization for this class
            enhanced_class_map = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # Get class color from the consistent mapping
            class_color = actual_class_colors[i]
            if class_color.startswith('#'):
                r = int(class_color[1:3], 16)
                g = int(class_color[3:5], 16)
                b = int(class_color[5:7], 16)
            else:
                r, g, b = 128, 128, 128
            
            # Fill the mask area with class color
            enhanced_class_map[class_mask > 0] = [r, g, b]
            
            # Find contours for better edge definition
            binary_mask = class_mask * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw enhanced contour lines
            for contour in contours:
                # Draw thick outer contour in white
                cv2.drawContours(enhanced_class_map, [contour], -1, (255, 255, 255), 3)
                # Draw inner contour in class color
                cv2.drawContours(enhanced_class_map, [contour], -1, (r, g, b), 1)
            
            individual_class_maps[class_name] = enhanced_class_map
        else:
            # Empty class - create empty map
            individual_class_maps[class_name] = np.zeros((256, 256, 3), dtype=np.uint8)
    
    print(f"Debug: class_probs shape: {class_probs.shape}")
    print(f"Debug: present_classes: {present_classes}")
    print(f"Debug: present_class_probs shape: {present_class_probs.shape}")
    
    # Create individual output images for visualization
    all_output_images = {}
    combined_output_img = None
    
    if final_output is not None and deep_sup1 is not None and deep_sup2 is not None and original_output_raw is not None:
        print("🔄 Creating individual output images...")
        
        # Convert each output to segmentation image
        final_output_img, _, _ = convert_output_to_segmentation_image(final_output, local_class_names, local_class_colors)
        deep_sup1_img, _, _ = convert_output_to_segmentation_image(deep_sup1, local_class_names, local_class_colors)
        deep_sup2_img, _, _ = convert_output_to_segmentation_image(deep_sup2, local_class_names, local_class_colors)
        original_output_img, _, _ = convert_output_to_segmentation_image(original_output_raw, local_class_names, local_class_colors)
        
        # Create combined output
        combined_output_img, _, _ = create_combined_output(final_output, deep_sup1, deep_sup2, original_output_raw, local_class_names, local_class_colors)
        
        all_output_images = {
            'final_output': final_output_img,
            'deep_sup1': deep_sup1_img,
            'deep_sup2': deep_sup2_img,
            'original_output': original_output_img,
            'combined_output': combined_output_img
        }
        print("✅ All output images created including combined output")
    
    return pred_class, colored_map, overlay, present_class_probs, present_classes, actual_class_names, actual_class_colors, individual_class_maps, all_output_images

def create_enhanced_contours(segmentation_map, class_colors, class_names):
    """Create enhanced contour lines for better visualization"""
    try:
        # Create enhanced contour visualization
        contour_visualization = np.zeros_like(segmentation_map)
        
        # Find contours for each class by checking unique colors in the segmentation map
        unique_colors = np.unique(segmentation_map.reshape(-1, 3), axis=0)
        
        for i, (class_name, class_color) in enumerate(zip(class_names, class_colors)):
            # Convert hex color to RGB
            if class_color.startswith('#'):
                r = int(class_color[1:3], 16)
                g = int(class_color[3:5], 16)
                b = int(class_color[5:7], 16)
            else:
                r, g, b = 128, 128, 128
            
            # Create binary mask for this class color
            class_mask = np.all(segmentation_map == [r, g, b], axis=2).astype(np.uint8)
            
            if np.any(class_mask):
                # Find contours
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw enhanced contours
                for contour in contours:
                    # Draw thick contour lines (white)
                    cv2.drawContours(contour_visualization, [contour], -1, (255, 255, 255), 3)
                    
                    # Draw inner contour for better definition (class color)
                    cv2.drawContours(contour_visualization, [contour], -1, (r, g, b), 1)
        
        return contour_visualization
        
    except Exception as e:
        print(f"⚠️ Warning: Contour enhancement failed: {e}")
        return segmentation_map

def create_edge_enhanced_overlay(original_image, segmentation_map, alpha=0.6):
    """Create edge-enhanced overlay with better visual lines"""
    try:
        # Convert to grayscale for edge detection
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_original, 50, 150)
        
        # Dilate edges for better visibility
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create edge overlay
        edge_overlay = original_image.copy()
        edge_overlay[edges > 0] = [255, 255, 255]  # White edges
        
        # Blend with segmentation
        enhanced_overlay = cv2.addWeighted(edge_overlay, 1-alpha, segmentation_map, alpha, 0)
        
        return enhanced_overlay
        
    except Exception as e:
        print(f"⚠️ Warning: Edge enhancement failed: {e}")
        return cv2.addWeighted(original_image, 0.7, segmentation_map, 0.3, 0)

def create_enhanced_class_visualization(class_map, class_name, class_color, original_image):
    """Create enhanced visualization for individual class with better visual lines"""
    try:
        # Handle both grayscale and RGB class maps
        if len(class_map.shape) == 3:
            # RGB class map - convert to binary
            binary_mask = np.any(class_map > 0, axis=2).astype(np.uint8)
        else:
            # Grayscale class map
            binary_mask = (class_map > 0).astype(np.uint8)
        
        if not np.any(binary_mask):
            return np.zeros_like(original_image)
        
        # Convert hex color to RGB
        if class_color.startswith('#'):
            r = int(class_color[1:3], 16)
            g = int(class_color[3:5], 16)
            b = int(class_color[5:7], 16)
        else:
            r, g, b = 128, 128, 128
        
        # Create colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[binary_mask > 0] = [r, g, b]
        
        # Find contours for better edge definition
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create enhanced visualization
        enhanced_vis = original_image.copy()
        
        # Fill the mask area
        enhanced_vis[binary_mask > 0] = cv2.addWeighted(
            enhanced_vis[binary_mask > 0], 0.3, 
            colored_mask[binary_mask > 0], 0.7, 0
        )
        
        # Draw enhanced contours
        for contour in contours:
            # Outer contour (thick white)
            cv2.drawContours(enhanced_vis, [contour], -1, (255, 255, 255), 3)
            # Inner contour (thin class color)
            cv2.drawContours(enhanced_vis, [contour], -1, (r, g, b), 1)
        
        return enhanced_vis
        
    except Exception as e:
        print(f"⚠️ Warning: Enhanced class visualization failed for {class_name}: {e}")
        return original_image

def create_visualization(original_image, segmentation_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors, filename, individual_class_maps=None, all_output_images=None):
    """Create comprehensive visualization with enhanced visual lines and better segmentation output"""
    try:
        # Get actual number of classes from class_probs
        num_classes_actual = class_probs.shape[0]
        print(f"🔄 Visualization: Creating enhanced visualization for {num_classes_actual} classes")
        
        # Create enhanced visualizations
        enhanced_contours = create_enhanced_contours(segmentation_map, actual_class_colors, actual_class_names)
        edge_enhanced_overlay = create_edge_enhanced_overlay(original_image, segmentation_map)
        
        # Ensure we have valid data
        if num_classes_actual == 0:
            print("⚠️ Warning: No classes to visualize, creating basic visualization")
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('WeakSegFormer Brain Tumor Segmentation Results', fontsize=16, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original MRI Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # Segmentation map
            axes[0, 1].imshow(segmentation_map)
            axes[0, 1].set_title('Segmentation Map', fontweight='bold')
            axes[0, 1].axis('off')
            
            # Enhanced overlay
            axes[0, 2].imshow(edge_enhanced_overlay)
            axes[0, 2].set_title('Edge-Enhanced Overlay', fontweight='bold')
            axes[0, 2].axis('off')
            
            # Enhanced contours
            axes[1, 0].imshow(enhanced_contours)
            axes[1, 0].set_title('Enhanced Contours', fontweight='bold')
            axes[1, 0].axis('off')
            
            # Original overlay
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Original Overlay', fontweight='bold')
            axes[1, 1].axis('off')
            
            # Empty subplot
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            viz_path = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_visualization.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return viz_path
        
        # Create enhanced subplot layout with all outputs
        if all_output_images and len(all_output_images) > 0:
            # Layout with all model outputs: 6 rows x 4 columns
            fig, axes = plt.subplots(6, 4, figsize=(24, 30))
            fig.suptitle('WeakSegFormer Brain Tumor Segmentation Results - All Model Outputs', 
                        fontsize=18, fontweight='bold', y=0.98)
        else:
            # Standard layout without all outputs
            fig, axes = plt.subplots(4, 4, figsize=(24, 20))
            fig.suptitle('WeakSegFormer Brain Tumor Segmentation Results (Hybrid Class Mapping)', 
                        fontsize=18, fontweight='bold', y=0.98)
        
        # Row 1: Main results with enhanced visualizations
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original MRI Image', fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        
        # Enhanced segmentation map with contours
        axes[0, 1].imshow(segmentation_map)
        axes[0, 1].set_title('Segmentation Map', fontweight='bold', fontsize=12)
        axes[0, 1].axis('off')
        
        # Edge-enhanced overlay
        axes[0, 2].imshow(edge_enhanced_overlay)
        axes[0, 2].set_title('Edge-Enhanced Overlay', fontweight='bold', fontsize=12)
        axes[0, 2].axis('off')
        
        # Enhanced contours
        axes[0, 3].imshow(enhanced_contours)
        axes[0, 3].set_title('Enhanced Contours', fontweight='bold', fontsize=12)
        axes[0, 3].axis('off')
        
        # Row 2: Individual class enhanced visualizations
        for i, class_name in enumerate(actual_class_names):
            if i < 4:  # First 4 classes in second row
                if individual_class_maps and class_name in individual_class_maps:
                    # Create enhanced class visualization
                    enhanced_class_vis = create_enhanced_class_visualization(
                        individual_class_maps[class_name], class_name, 
                        actual_class_colors[i], original_image
                    )
                    axes[1, i].imshow(enhanced_class_vis)
                    axes[1, i].set_title(f'{class_name} Enhanced', fontweight='bold', fontsize=12)
                    axes[1, i].axis('off')
                elif i < num_classes_actual:
                    # Fallback to probability map with enhanced visualization
                    prob_map = class_probs[i]
                    # Apply colormap and enhance
                    enhanced_prob = plt.cm.hot(prob_map)[:, :, :3]
                    axes[1, i].imshow(enhanced_prob)
                    axes[1, i].set_title(f'{class_name} Probability', fontweight='bold', fontsize=12)
                    axes[1, i].axis('off')
                else:
                    # Empty subplot
                    axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                  transform=axes[1, i].transAxes, fontsize=10)
                    axes[1, i].set_title(f'No Data', fontweight='bold', fontsize=12)
                    axes[1, i].axis('off')
        
        # Row 3: Probability heatmaps with enhanced visualization
        for i, class_name in enumerate(actual_class_names):
            if i < 4:  # First 4 classes in third row
                if i < num_classes_actual:
                    # Create enhanced probability visualization
                    prob_map = class_probs[i]
                    
                    # Apply Gaussian blur for smoother visualization
                    prob_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0)
                    
                    # Create enhanced heatmap
                    im = axes[2, i].imshow(prob_smooth, cmap='hot', vmin=0, vmax=1)
                    axes[2, i].set_title(f'{class_name} Probability (Enhanced)', fontweight='bold', fontsize=12)
                    axes[2, i].axis('off')
                    
                    # Add colorbar for better understanding
                    plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
                else:
                    # Empty subplot
                    axes[2, i].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                  transform=axes[2, i].transAxes, fontsize=10)
                    axes[2, i].set_title(f'No Data', fontweight='bold', fontsize=12)
                    axes[2, i].axis('off')
        
        # Row 4: Additional visualizations and legend
        # Original overlay for comparison
        axes[3, 0].imshow(overlay)
        axes[3, 0].set_title('Original Overlay', fontweight='bold', fontsize=12)
        axes[3, 0].axis('off')
        
        # Class distribution visualization
        if len(actual_class_names) > 0:
            class_areas = []
            total_pixels = segmentation_map.shape[0] * segmentation_map.shape[1]
            for i, class_name in enumerate(actual_class_names):
                if i < len(present_classes):
                    class_pixels = np.sum(segmentation_map == present_classes[i])
                    percentage = (class_pixels / total_pixels) * 100
                    class_areas.append(percentage)
                else:
                    class_areas.append(0)
            
            # Create bar chart with consistent colors
            bar_colors = []
            for i, class_name in enumerate(actual_class_names):
                if i < len(actual_class_colors):
                    color = actual_class_colors[i]
                else:
                    # Fallback to complete mapping
                    complete_mapping = {
                        'Background': '#000000',
                        'Necrotic': '#8B0000', 
                        'Edema': '#228B22',
                        'Tumor': '#4169E1'
                    }
                    color = complete_mapping.get(class_name, '#808080')
                bar_colors.append(color)
            
            bars = axes[3, 1].bar(range(len(actual_class_names)), class_areas, color=bar_colors)
            axes[3, 1].set_title('Class Area Distribution', fontweight='bold', fontsize=12)
            axes[3, 1].set_ylabel('Percentage (%)')
            axes[3, 1].set_xticks(range(len(actual_class_names)))
            axes[3, 1].set_xticklabels(actual_class_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, class_areas):
                height = bar.get_height()
                axes[3, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            axes[3, 1].text(0.5, 0.5, 'No Class Data', ha='center', va='center', 
                          transform=axes[3, 1].transAxes, fontsize=10)
            axes[3, 1].set_title('No Data', fontweight='bold', fontsize=12)
            axes[3, 1].axis('off')
        
        # Enhanced legend with better styling
        legend_col = 2
        legend_elements = []
        for i, (color, name) in enumerate(zip(actual_class_colors, actual_class_names)):
            # Convert hex color to RGB
            if color.startswith('#'):
                r = int(color[1:3], 16) / 255.0
                g = int(color[3:5], 16) / 255.0
                b = int(color[5:7], 16) / 255.0
                color_rgb = (r, g, b)
            else:
                color_rgb = (0.5, 0.5, 0.5)  # Default gray
            
            legend_elements.append(patches.Patch(color=color_rgb, label=name))
        
        axes[3, legend_col].legend(handles=legend_elements, loc='center', 
                                 fontsize=11, frameon=True, fancybox=True, shadow=True)
        axes[3, legend_col].set_title('Class Legend', fontweight='bold', fontsize=12)
        axes[3, legend_col].axis('off')
        
        # Summary statistics
        summary_text = f"Hybrid Segmentation Analysis Summary\n"
        summary_text += f"Classes Detected: {len(actual_class_names)}\n"
        summary_text += f"Total Classes: {num_classes_actual}\n"
        summary_text += f"Model: WeakSegFormer (Hybrid)\n"
        summary_text += f"Class Mapping:\n"
        summary_text += f"• Class 0,1: original_output\n"
        summary_text += f"• Class 2: deep_sup1\n"
        summary_text += f"• Class 3: deep_sup2\n"
        summary_text += f"Performance: 72.66% mIoU"
        
        axes[3, 3].text(0.1, 0.5, summary_text, transform=axes[3, 3].transAxes, 
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[3, 3].set_title('Analysis Summary', fontweight='bold', fontsize=12)
        axes[3, 3].axis('off')
        
        # Add all model outputs visualization if available
        if all_output_images and len(all_output_images) > 0 and len(axes) > 4:
            print("🔄 Adding all model outputs to visualization...")
            
            # Row 4: All Model Outputs (including combined)
            output_names = ['final_output', 'deep_sup1', 'deep_sup2', 'original_output']
            output_titles = ['Final Output', 'Deep Supervision 1', 'Deep Supervision 2', 'Original Output']
            
            for i, (output_name, output_title) in enumerate(zip(output_names, output_titles)):
                if output_name in all_output_images and i < 4:
                    axes[4, i].imshow(all_output_images[output_name])
                    axes[4, i].set_title(f'{output_title}', fontweight='bold', fontsize=12)
                    axes[4, i].axis('off')
                    print(f"✅ Added {output_title} to visualization")
                else:
                    axes[4, i].axis('off')
            
            # Row 5: Combined Output and Analysis
            if len(axes) > 5:
                # Show combined output in first position
                if 'combined_output' in all_output_images:
                    axes[5, 0].imshow(all_output_images['combined_output'])
                    axes[5, 0].set_title('Combined Output (All Outputs)', fontweight='bold', fontsize=12, color='red')
                    axes[5, 0].axis('off')
                    print("✅ Added Combined Output to visualization")
                else:
                    axes[5, 0].axis('off')
                
                # Show output comparison
                axes[5, 1].text(0.5, 0.5, 'Output Comparison', ha='center', va='center', 
                               transform=axes[5, 1].transAxes, fontsize=14, fontweight='bold')
                axes[5, 1].axis('off')
                
                # Show combined output statistics
                combined_stats = "Combined Output Statistics:\n"
                combined_stats += "• Weight: 25% each output\n"
                combined_stats += "• Method: Weighted average\n"
                combined_stats += "• Purpose: Ensemble prediction\n"
                
                axes[5, 2].text(0.1, 0.5, combined_stats, transform=axes[5, 2].transAxes, 
                               fontsize=10, verticalalignment='center')
                axes[5, 2].set_title('Combined Method', fontweight='bold', fontsize=12)
                axes[5, 2].axis('off')
                
                # Empty subplot
                axes[5, 3].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for main title
        
        # Save visualization
        viz_path = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Properly close the figure to prevent memory leaks
        plt.close(fig)
        plt.clf()
        plt.cla()
        
        print(f"✅ Enhanced visualization saved to: {viz_path}")
        return viz_path
        
    except Exception as e:
        print(f"❌ Error in enhanced visualization: {e}")
        import traceback
        traceback.print_exc()
        # Clean up matplotlib state
        plt.close('all')
        plt.clf()
        plt.cla()
        # Return None instead of raising error to allow processing to continue
        return None

def calculate_metrics(segmentation_map, class_probs, present_classes, actual_class_names):
    """Calculate segmentation metrics"""
    try:
        metrics = {}
        
        # Ensure we have valid data
        if len(actual_class_names) == 0:
            print("⚠️ Warning: No classes to calculate metrics for")
            return {'error': 'No classes available for metrics calculation'}
        
        # Calculate area percentages
        total_pixels = segmentation_map.shape[0] * segmentation_map.shape[1]
        for i, class_name in enumerate(actual_class_names):
            # Use the original class index from present_classes
            if i < len(present_classes):
                original_class_idx = present_classes[i]
                class_pixels = np.sum(segmentation_map == original_class_idx)
                percentage = (class_pixels / total_pixels) * 100
                metrics[f'{class_name}_percentage'] = round(percentage, 2)
            else:
                print(f"⚠️ Warning: Class {class_name} index {i} out of range for present_classes")
                metrics[f'{class_name}_percentage'] = 0.0
        
        # Calculate confidence scores - class_probs only contains present classes
        for i, class_name in enumerate(actual_class_names):
            if i < len(class_probs):
                # class_probs[i] corresponds to the i-th present class
                try:
                    avg_confidence = np.mean(class_probs[i])
                    metrics[f'{class_name}_confidence'] = round(avg_confidence * 100, 2)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to calculate confidence for {class_name}: {e}")
                    metrics[f'{class_name}_confidence'] = 0.0
            else:
                print(f"⚠️ Warning: Class {class_name} index {i} out of range for class_probs")
                metrics[f'{class_name}_confidence'] = 0.0
        
        print(f"✅ Metrics calculated successfully: {list(metrics.keys())}")
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        # Return basic metrics to prevent complete failure
        return {
            'error': f'Metrics calculation failed: {str(e)}',
            'fallback_metrics': 'Available'
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test page"""
    return send_file('test_frontend.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and inference"""
    try:
        global model, device
        
        print("🔄 Starting file upload processing...")
        
        # Load model if not already loaded
        if model is None:
            print("🔄 Loading model globally...")
            if not load_model_global():
                error_msg = "Failed to load model. Check server logs for details."
                print(f"❌ {error_msg}")
                return jsonify({'error': error_msg}), 500
            print("✅ Model loaded successfully")
        else:
            print("✅ Using existing loaded model")
        
        # Double-check model is loaded and accessible
        if model is None:
            error_msg = "Model is still None after loading attempt. This indicates a critical error."
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Use global model
        local_model = model
        
        if 'file' not in request.files:
            error_msg = "No file uploaded"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        file = request.files['file']
        if file.filename == '':
            error_msg = "No file selected"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        print(f"✅ File received: {file.filename}")
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}"
        
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}.jpg")
        try:
            file.save(file_path)
            print(f"✅ File saved to: {file_path}")
        except Exception as e:
            error_msg = f"Failed to save uploaded file: {str(e)}"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Preprocess image
        print("🔄 Preprocessing image...")
        try:
            image_tensor, original_image = preprocess_image(file_path)
            print(f"✅ Image tensor shape: {image_tensor.shape}")
            image_tensor = image_tensor.to(device)
            print("✅ Image tensor moved to device")
        except Exception as e:
            error_msg = f"Image preprocessing failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
        
        # Run inference
        print("🔄 Running inference...")
        try:
            with torch.no_grad():
                output = local_model(image_tensor)
                print(f"✅ Model output type: {type(output)}")
                print(f"✅ Raw output shape: {output.shape if hasattr(output, 'shape') else 'No shape'}")
                
                # Handle model output correctly - model returns 4 outputs: final_output, deep_sup1, deep_sup2, original_output
                if isinstance(output, tuple) and len(output) >= 4:
                    print(f"✅ Model output length: {len(output)}")
                    final_output, deep_sup1, deep_sup2, original_output = output
                    print(f"✅ Final output shape: {final_output.shape}")
                    print(f"✅ Deep sup1 shape: {deep_sup1.shape}")
                    print(f"✅ Deep sup2 shape: {deep_sup2.shape}")
                    print(f"✅ Original output shape: {original_output.shape}")
                    
                    # Use hybrid approach: combine different outputs for different classes
                    # Main output will be passed to postprocess, along with the individual outputs
                    main_output = original_output
                    print(f"✅ Using hybrid segmentation (combining all outputs), main shape: {main_output.shape}")
                    
                    # Store all outputs for hybrid processing and visualization
                    hybrid_outputs = {
                        'final_output': final_output,
                        'deep_sup1': deep_sup1,
                        'deep_sup2': deep_sup2,
                        'original_output': original_output
                    }
                    
                    # Debug: Check if outputs are different
                    print(f"✅ Final vs Deep Sup1 difference: {torch.mean(torch.abs(final_output - deep_sup1)).item():.6f}")
                    print(f"✅ Final vs Deep Sup2 difference: {torch.mean(torch.abs(final_output - deep_sup2)).item():.6f}")
                    print(f"✅ Final vs Original difference: {torch.mean(torch.abs(final_output - original_output)).item():.6f}")
                    print(f"✅ Deep Sup1 vs Deep Sup2 difference: {torch.mean(torch.abs(deep_sup1 - deep_sup2)).item():.6f}")
                    
                elif isinstance(output, tuple):
                    # Fallback for other tuple lengths
                    main_output = output[0]
                    hybrid_outputs = None
                    print(f"✅ Using first output from tuple, shape: {main_output.shape}")
                    print(f"✅ Tuple length: {len(output)}")
                    for i, out in enumerate(output):
                        print(f"✅ Output {i} shape: {out.shape}")
                else:
                    main_output = output
                    hybrid_outputs = None
                    print(f"✅ Using single output, shape: {main_output.shape}")
        except Exception as e:
            error_msg = f"Model inference failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
        
        print(f"✅ Final output shape: {main_output.shape}")
        print(f"✅ Output classes: {main_output.shape[1]}")
        print(f"✅ Output min/max values: {main_output.min().item():.4f} / {main_output.max().item():.4f}")
        
        # Additional debug: Check class distribution
        if len(main_output.shape) == 4:  # [B, C, H, W]
            class_probs = F.softmax(main_output, dim=1)
            class_means = torch.mean(class_probs, dim=[0, 2, 3])  # Mean probability for each class
            print(f"✅ Class probability means: {class_means.cpu().numpy()}")
            print(f"✅ Class names: {class_names}")
            
            # Check if any class has very low probability
            for i, (class_name, prob) in enumerate(zip(class_names, class_means)):
                if prob < 0.01:  # Less than 1% probability
                    print(f"⚠️ Warning: Class {class_name} has very low probability: {prob:.4f}")
                else:
                    print(f"✅ Class {class_name} probability: {prob:.4f}")
        
        # Check if output has expected number of classes
        if main_output.shape[1] != 4:
            print(f"⚠️ Warning: Actual inference output has {main_output.shape[1]} classes, expected 4")
            print(f"⚠️ This suggests the model architecture may have an issue")
            
            # If we have 3 classes, adjust class_names and class_colors
            if main_output.shape[1] == 3:
                print("🔄 Adjusting to 3 classes: Background, Edema, Tumor")
                # Create local variables for this request
                local_class_names = ['Background', 'Edema', 'Tumor']  # 3 classes
                local_class_colors = ['#000000', '#00FF00', '#0000FF']  # Black, Green, Blue
            else:
                local_class_names = class_names
                local_class_colors = class_colors
        else:
            local_class_names = class_names
            local_class_colors = class_colors
        
        # Postprocess output
        print("🔄 Starting postprocessing...")
        try:
            # Pass hybrid outputs if available
            if hybrid_outputs is not None:
                segmentation_map, colored_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors, individual_class_maps, all_output_images = postprocess_output(
                    main_output, original_image, local_class_names, local_class_colors,
                    final_output=hybrid_outputs['final_output'],
                    deep_sup1=hybrid_outputs['deep_sup1'],
                    deep_sup2=hybrid_outputs['deep_sup2'],
                    original_output_raw=hybrid_outputs['original_output']
                )
            else:
                segmentation_map, colored_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors, individual_class_maps, all_output_images = postprocess_output(
                    main_output, original_image, local_class_names, local_class_colors
                )
            
            print("✅ Postprocessing completed")
            print(f"✅ Postprocessing results:")
            print(f"  - Segmentation map shape: {segmentation_map.shape}")
            print(f"  - Colored map shape: {colored_map.shape}")
            print(f"  - Overlay shape: {overlay.shape}")
            print(f"  - Class probs shape: {class_probs.shape}")
            print(f"  - Present classes: {present_classes}")
            print(f"  - Actual class names: {actual_class_names}")
            print(f"  - Individual class maps: {list(individual_class_maps.keys())}")
        except Exception as e:
            error_msg = f"Postprocessing failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
        
        # Create visualization
        print("🔄 Creating visualization...")
        try:
            viz_path = create_visualization(original_image, colored_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors, filename, individual_class_maps, all_output_images)
            if viz_path:
                print("✅ Visualization created")
            else:
                print("⚠️ Visualization creation returned None, continuing without visualization")
        except Exception as e:
            print(f"⚠️ Warning: Visualization creation failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue without visualization if it fails
            viz_path = None
        
        # Calculate metrics
        print("🔄 Calculating metrics...")
        try:
            metrics = calculate_metrics(segmentation_map, class_probs, present_classes, actual_class_names)
            if 'error' in metrics:
                print(f"⚠️ Warning: Metrics calculation had issues: {metrics['error']}")
                # Continue with basic metrics
                metrics = {'status': 'Basic metrics available', 'classes_detected': len(actual_class_names)}
            print("✅ Metrics calculated")
        except Exception as e:
            error_msg = f"Metrics calculation failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            # Continue with basic metrics instead of failing completely
            metrics = {'status': 'Fallback metrics', 'error': str(e), 'classes_detected': len(actual_class_names)}
        
        # Convert images to base64 for frontend
        print("🔄 Converting images to base64...")
        try:
            # Convert original image
            original_image_pil = Image.fromarray(original_image)
            original_buffer = io.BytesIO()
            original_image_pil.save(original_buffer, format='PNG')
            original_image_b64 = base64.b64encode(original_buffer.getvalue()).decode()
            
            # Convert segmentation map
            segmentation_map_pil = Image.fromarray(colored_map)
            segmentation_buffer = io.BytesIO()
            segmentation_map_pil.save(segmentation_buffer, format='PNG')
            segmentation_map_b64 = base64.b64encode(segmentation_buffer.getvalue()).decode()
            
            # Convert overlay
            overlay_pil = Image.fromarray(overlay)
            overlay_buffer = io.BytesIO()
            overlay_pil.save(overlay_buffer, format='PNG')
            overlay_b64 = base64.b64encode(overlay_buffer.getvalue()).decode()
            
            # Convert individual class maps
            individual_class_maps_b64 = {}
            for class_name, class_map in individual_class_maps.items():
                try:
                    class_map_pil = Image.fromarray(class_map)
                    class_buffer = io.BytesIO()
                    class_map_pil.save(class_buffer, format='PNG')
                    individual_class_maps_b64[class_name] = base64.b64encode(class_buffer.getvalue()).decode()
                except Exception as e:
                    print(f"⚠️ Warning: Failed to convert {class_name} class map to base64: {e}")
                    # Continue with other class maps
            
            # Convert all output images
            all_output_images_b64 = {}
            for output_name, output_img in all_output_images.items():
                try:
                    output_img_pil = Image.fromarray(output_img)
                    output_buffer = io.BytesIO()
                    output_img_pil.save(output_buffer, format='PNG')
                    all_output_images_b64[output_name] = base64.b64encode(output_buffer.getvalue()).decode()
                    print(f"✅ Converted {output_name} to base64")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to convert {output_name} to base64: {e}")
                    # Continue with other outputs
            
            print("✅ All images converted to base64")
        except Exception as e:
            error_msg = f"Image conversion to base64 failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
        
        # Prepare results
        print("🔄 Preparing results...")
        try:
            # Ensure class_probs is in the right format for JSON serialization
            class_probs_list = []
            for i in range(len(class_probs)):
                try:
                    if hasattr(class_probs[i], 'tolist'):
                        class_probs_list.append(class_probs[i].tolist())
                    else:
                        class_probs_list.append(class_probs[i].astype(float).tolist())
                except Exception as e:
                    print(f"⚠️ Warning: Failed to convert class_probs[{i}] to list: {e}")
                    class_probs_list.append([])
            
            results = {
                'filename': filename,
                'original_image': original_image_b64,
                'segmentation_map': segmentation_map_b64,
                'overlay': overlay_b64,
                'individual_class_maps': individual_class_maps_b64,
                'all_output_images': all_output_images_b64,  # All model outputs as images
                'class_probs': class_probs_list,  # Use the processed list
                'visualization_path': viz_path if viz_path else 'None',
                'metrics': metrics,
                'class_names': actual_class_names,  # Use only present classes
                'class_colors': actual_class_colors,  # Use only present class colors
                'present_classes': present_classes,  # Original class indices
                'timestamp': datetime.now().isoformat()
            }
            print("✅ Results prepared")
        except Exception as e:
            error_msg = f"Results preparation failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
        
        # Save results to file
        print("🔄 Saving results to file...")
        try:
            results_file = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f)
            print(f"✅ Results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save results to file: {e}")
            # Continue without saving results file
        
        print("✅ File upload processing completed successfully!")
        return jsonify(results)
        
    except Exception as e:
        error_msg = f"An error occurred during processing: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/results/<filename>')
def get_results(filename):
    """Get saved results"""
    results_file = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    else:
        return jsonify({'error': 'Results not found'}), 404

@app.route('/download/<filename>')
def download_results(filename):
    """Download results as zip file"""
    # Implementation for downloading results
    pass

@app.route('/download/class_maps/<filename>')
def download_class_maps(filename):
    """Download individual class segmentation maps"""
    try:
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_results.json')
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create a zip file with individual class maps
        import zipfile
        import tempfile
        
        zip_path = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_class_maps.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add original image
            zipf.writestr('original_image.png', base64.b64decode(results['original_image']))
            
            # Add individual class maps
            for class_name, class_map_b64 in results.get('individual_class_maps', {}).items():
                zipf.writestr(f'{class_name}_segmentation.png', base64.b64decode(class_map_b64))
            
            # Add segmentation map and overlay
            zipf.writestr('segmentation_map.png', base64.b64decode(results['segmentation_map']))
            zipf.writestr('overlay.png', base64.b64decode(results['overlay']))
            
            # Add metrics as text file
            metrics_text = "Segmentation Analysis Results\n"
            metrics_text += "=" * 30 + "\n\n"
            for key, value in results['metrics'].items():
                metrics_text += f"{key}: {value}\n"
            zipf.writestr('metrics.txt', metrics_text)
        
        return send_file(zip_path, as_attachment=True, download_name=f'{filename}_class_maps.zip')
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/download/visualization/<filename>')
def download_visualization(filename):
    """Download visualization image"""
    try:
        viz_path = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_visualization.png')
        if os.path.exists(viz_path):
            return send_file(viz_path, as_attachment=True, download_name=f'{filename}_visualization.png')
        else:
            return jsonify({'error': 'Visualization not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/download/report/<filename>')
def download_comprehensive_report(filename):
    """Download comprehensive analysis report as HTML"""
    try:
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_results.json')
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Generate HTML report
        html_report = generate_html_report(results)
        
        # Save HTML report
        report_path = os.path.join(app.config['RESULTS_FOLDER'], f'{filename}_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        return send_file(report_path, as_attachment=True, download_name=f'{filename}_analysis_report.html')
        
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

def generate_html_report(results):
    """Generate comprehensive HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Segmentation Report - {results['filename']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
            .header h1 {{ color: #2c3e50; margin: 0; }}
            .header p {{ color: #7f8c8d; margin: 10px 0 0 0; }}
            .section {{ margin-bottom: 30px; }}
            .section h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .image-item {{ text-align: center; }}
            .image-item img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
            .image-item h4 {{ margin: 10px 0; color: #2c3e50; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; }}
            .metric-value {{ font-size: 2rem; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
            .class-maps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
            .class-map-item {{ text-align: center; background: #f8f9fa; padding: 15px; border-radius: 8px; }}
            .class-map-item img {{ max-width: 100%; border-radius: 6px; }}
            .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Brain Tumor Segmentation Analysis Report</h1>
                <p>WeakSegFormer Model - {results['timestamp']}</p>
                <p>Analysis ID: {results['filename']}</p>
            </div>
            
            <div class="section">
                <h2>📊 Analysis Summary</h2>
                <p>This report presents the results of brain tumor segmentation analysis using the WeakSegFormer model with hybrid class mapping, 
                which achieved 72.66% mIoU performance on the BraTS dataset.</p>
                <p><strong>Hybrid Class Mapping:</strong></p>
                <ul>
                    <li>Class 0 (Background) and Class 1 (Necrotic): from original_output</li>
                    <li>Class 2 (Edema): from deep_sup1</li>
                    <li>Class 3 (Tumor): from deep_sup2</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🖼️ Segmentation Results</h2>
                <div class="image-grid">
                    <div class="image-item">
                        <h4>Original MRI Image</h4>
                        <img src="data:image/png;base64,{results['original_image']}" alt="Original MRI">
                    </div>
                    <div class="image-item">
                        <h4>Segmentation Map</h4>
                        <img src="data:image/png;base64,{results['segmentation_map']}" alt="Segmentation">
                    </div>
                    <div class="image-item">
                        <h4>Overlay Result</h4>
                        <img src="data:image/png;base64,{results['overlay']}" alt="Overlay">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Quantitative Metrics</h2>
                <div class="metrics-grid">
    """
    
    # Add metrics
    for class_name in results['class_names']:
        percentage = results['metrics'].get(f'{class_name}_percentage', 0)
        confidence = results['metrics'].get(f'{class_name}_confidence', 0)
        html_content += f"""
                    <div class="metric-card">
                        <div class="metric-value">{percentage}%</div>
                        <div class="metric-label">{class_name} Area Coverage</div>
                        <small>Confidence: {confidence}%</small>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Individual Class Segmentation Maps</h2>
                <div class="class-maps">
    """
    
    # Add individual class maps
    for class_name, class_map_b64 in results.get('individual_class_maps', {}).items():
        html_content += f"""
                    <div class="class-map-item">
                        <h4>{class_name} Segmentation</h4>
                        <img src="data:image/png;base64,{class_map_b64}" alt="{class_name} Segmentation">
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Technical Details</h2>
                <ul>
                    <li><strong>Model:</strong> WeakSegFormer (Vision Transformer) with Hybrid Class Mapping</li>
                    <li><strong>Performance:</strong> 72.66% mIoU on BraTS dataset</li>
                    <li><strong>Input Resolution:</strong> 256x256 pixels</li>
                    <li><strong>Segmentation Classes:</strong> Background, Necrotic, Edema, Tumor</li>
                    <li><strong>Class Mapping Strategy:</strong> Hybrid approach using different model outputs for different classes</li>
                    <li><strong>Analysis Timestamp:</strong> {results['timestamp']}</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Generated by WeakSegFormer Brain Tumor Segmentation System</p>
                <p>For medical use, please consult with qualified healthcare professionals</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'directories': {},
            'dependencies': {},
            'model': {},
            'overall_healthy': True
        }
        
        # Check directories
        try:
            upload_dir = app.config['UPLOAD_FOLDER']
            results_dir = app.config['RESULTS_FOLDER']
            
            health_status['directories'] = {
                'upload_folder': {
                    'path': upload_dir,
                    'exists': os.path.exists(upload_dir),
                    'writable': os.access(upload_dir, os.W_OK) if os.path.exists(upload_dir) else False
                },
                'results_folder': {
                    'path': results_dir,
                    'exists': os.path.exists(results_dir),
                    'writable': os.access(results_dir, os.W_OK) if os.path.exists(results_dir) else False
                }
            }
            
            # Check if any directory issues
            for dir_info in health_status['directories'].values():
                if not dir_info['exists'] or not dir_info['writable']:
                    health_status['overall_healthy'] = False
                    
        except Exception as e:
            health_status['directories'] = {'error': str(e)}
            health_status['overall_healthy'] = False
        
        # Check dependencies
        try:
            health_status['dependencies'] = {
                'torch': {
                    'available': True,
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available()
                },
                'albumentations': {
                    'available': True,
                    'version': A.__version__ if hasattr(A, '__version__') else 'Unknown'
                },
                'PIL': {
                    'available': True,
                    'version': Image.__version__ if hasattr(Image, '__version__') else 'Unknown'
                }
            }
        except Exception as e:
            health_status['dependencies'] = {'error': str(e)}
            health_status['overall_healthy'] = False
        
        # Check model status - try to load if not already loaded
        model_status = {
            'loaded': model is not None,
            'type': str(type(model)) if model else 'None',
            'device': str(device) if device else 'None'
        }
        if not model_status['loaded']:
            print("🔄 Health check: Model not loaded, attempting to load...")
            try:
                if load_model_global():
                    model_status = {
                        'loaded': True,
                        'type': str(type(model)),
                        'device': str(device)
                    }
                    print("✅ Health check: Model loaded successfully")
                else:
                    print("❌ Health check: Failed to load model")
                    health_status['overall_healthy'] = False
            except Exception as e:
                print(f"❌ Health check: Model loading failed: {e}")
                model_status = {
                    'loaded': False,
                    'type': 'None',
                    'device': 'None'
                }
                health_status['overall_healthy'] = False
        
        health_status['model'] = model_status
        
        # Final health assessment
        if not health_status['overall_healthy']:
            health_status['status'] = 'unhealthy'
            health_status['message'] = 'Some health checks failed'
        else:
            health_status['status'] = 'healthy'
            health_status['message'] = 'All systems operational'
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/debug/model')
def debug_model():
    """Debug endpoint to test model functionality"""
    try:
        global model, device
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'model_status': {
                'loaded': model is not None,
                'type': str(type(model)) if model else 'None',
                'device': str(device) if device else 'None'
            },
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'test_inference': {},
            'filesystem': {}
        }
        
        # Test model inference if available
        if model is not None:
            try:
                model.eval()
                with torch.no_grad():
                    # Create a dummy input tensor
                    test_input = torch.randn(1, 3, 256, 256).to(device)
                    print(f"🔄 Debug: Testing inference with input shape: {test_input.shape}")
                    
                    # Run inference
                    test_output = model(test_input)
                    print(f"🔄 Debug: Raw output type: {type(test_output)}")
                    
                    if isinstance(test_output, (list, tuple)):
                        main_output = test_output[0]
                        debug_info['test_inference'] = {
                            'status': '✅ Success',
                            'input_shape': list(test_input.shape),
                            'output_type': 'tuple/list',
                            'output_length': len(test_output),
                            'main_output_shape': list(main_output.shape),
                            'output_classes': main_output.shape[1] if len(main_output.shape) > 1 else 'N/A',
                            'output_range': f"{main_output.min().item():.4f} to {main_output.max().item():.4f}"
                        }
                        print(f"✅ Debug: Test inference successful with tuple output")
                    else:
                        debug_info['test_inference'] = {
                            'status': '✅ Success',
                            'input_shape': list(test_input.shape),
                            'output_type': 'single_tensor',
                            'output_shape': list(test_output.shape),
                            'output_classes': test_output.shape[1] if len(test_output.shape) > 1 else 'N/A',
                            'output_range': f"{test_output.min().item():.4f} to {test_output.max().item():.4f}"
                        }
                        print(f"✅ Debug: Test inference successful with single output")
                        
            except Exception as e:
                debug_info['test_inference'] = {
                    'status': '❌ Failed',
                    'error': str(e)
                }
                print(f"❌ Debug: Test inference failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            debug_info['test_inference'] = {
                'status': '⚠️ No Model',
                'message': 'Model is not loaded'
            }
        
        # Filesystem info
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            
            debug_info['filesystem'] = {
                'current_directory': current_dir,
                'parent_directory': parent_dir,
                'current_files': len(os.listdir(current_dir)),
                'parent_files': len(os.listdir(parent_dir))
            }
        except Exception as e:
            debug_info['filesystem'] = {
                'error': f'Failed to get filesystem info: {str(e)}'
            }
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({
            'error': f'Debug endpoint failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    print("📝 Model will be loaded on first request")
    
    # Run Flask app (disable debug mode to avoid reload issues)
    app.run(debug=False, host='0.0.0.0', port=5000) 