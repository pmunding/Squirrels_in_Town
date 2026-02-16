# -*- coding: utf-8 -*-
"""
YOLO Model Training Script for Local/VSCode Environment
Complete workflow for training YOLO11, YOLOv8, or YOLOv5 object detection models
Works on Windows, Linux, and macOS with local GPU or CPU
"""

import os
import yaml
import shutil
import zipfile
from pathlib import Path
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Set your paths here
DATASET_ZIP = "data.zip"  # Path to your zipped dataset
WORKSPACE = "yolo_training"  # Working directory
CLASSES_FILE = "classes.txt"  # Path to classes file (will be in extracted data)

# Training parameters
MODEL = "yolo11s.pt"  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11xl
EPOCHS = 60
IMG_SIZE = 640
BATCH_SIZE = 16  # Adjust based on your GPU memory (8, 16, 32)
DEVICE = 0  # 0 for GPU, 'cpu' for CPU

# ==============================================================================
# STEP 0: VERIFY GPU AVAILABILITY
# ==============================================================================

def check_gpu():
    """Check if GPU is available"""
    import torch
    print("\n" + "="*60)
    print("CHECKING SYSTEM CONFIGURATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: No GPU detected. Training will use CPU (much slower)")
    print("="*60 + "\n")

# ==============================================================================
# STEP 1: SETUP WORKSPACE
# ==============================================================================

def setup_workspace(workspace_dir):
    """Create workspace directory structure"""
    print(f"Setting up workspace at: {workspace_dir}")
    
    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (workspace_path / "data").mkdir(exist_ok=True)
    (workspace_path / "custom_data").mkdir(exist_ok=True)
    (workspace_path / "runs").mkdir(exist_ok=True)
    
    print(f"✓ Workspace created at {workspace_path.absolute()}\n")
    return workspace_path

# ==============================================================================
# STEP 2: EXTRACT AND PREPARE DATASET
# ==============================================================================

def extract_dataset(zip_path, extract_to):
    """Extract dataset from zip file"""
    print(f"Extracting dataset from {zip_path}...")
    
    if not os.path.exists(zip_path):
        print(f"ERROR: Dataset not found at {zip_path}")
        print("Please place your data.zip file in the current directory")
        return False
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✓ Dataset extracted to {extract_to}\n")
    return True

def split_dataset(data_path, output_path, train_ratio=0.9):
    """Split dataset into train and validation sets"""
    import random
    
    print(f"Splitting dataset (train: {train_ratio*100}%, val: {(1-train_ratio)*100}%)...")
    
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Find images and labels
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    if not images_dir.exists():
        print(f"ERROR: Images directory not found at {images_dir}")
        return False
    
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found at {labels_dir}")
        return False
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(ext)))
    
    if len(image_files) == 0:
        print("ERROR: No images found in images directory")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Train: {len(train_images)} images")
    print(f"Validation: {len(val_images)} images")
    
    # Create directory structure
    for split in ['train', 'validation']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_files(image_list, split_name):
        for img_path in image_list:
            # Copy image
            shutil.copy(img_path, output_path / split_name / 'images' / img_path.name)
            
            # Copy corresponding label
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / label_name
            if label_path.exists():
                shutil.copy(label_path, output_path / split_name / 'labels' / label_name)
    
    copy_files(train_images, 'train')
    copy_files(val_images, 'validation')
    
    print("✓ Dataset split complete\n")
    return True

# ==============================================================================
# STEP 3: CREATE TRAINING CONFIGURATION
# ==============================================================================

def create_config_file(classes_file, output_yaml, data_path):
    """Create YAML configuration file for training"""
    print("Creating training configuration...")
    
    if not os.path.exists(classes_file):
        print(f"ERROR: classes.txt not found at {classes_file}")
        print("Please create a classes.txt file with one class name per line")
        return False
    
    # Read class names
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create config dictionary
    config = {
        'path': str(Path(data_path).absolute()),
        'train': 'train/images',
        'val': 'validation/images',
        'nc': len(classes),
        'names': classes
    }
    
    # Write YAML file
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"✓ Config file created at {output_yaml}\n")
    
    # Display config
    print("Configuration:")
    print("-" * 40)
    with open(output_yaml, 'r') as f:
        print(f.read())
    print("-" * 40 + "\n")
    
    return True

# ==============================================================================
# STEP 4: TRAIN MODEL
# ==============================================================================

def train_model(config_file, model, epochs, img_size, batch_size, device, project_dir):
    """Train YOLO model"""
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Initialize model
    model_obj = YOLO(model)
    
    # Train
    results = model_obj.train(
        data=config_file,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project_dir,
        name='train'
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved at: {project_dir}/train/weights/best.pt")
    print(f"Training results: {project_dir}/train/results.png")
    print("="*60 + "\n")
    
    return results

# ==============================================================================
# STEP 5: TEST MODEL
# ==============================================================================

def test_model(model_path, source_dir, project_dir):
    """Run inference on validation images"""
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    
    model = YOLO(model_path)
    
    results = model.predict(
        source=source_dir,
        save=True,
        project=project_dir,
        name='predict'
    )
    
    print(f"✓ Predictions saved to: {project_dir}/predict")
    print("="*60 + "\n")
    
    return results

def display_results(predict_dir, num_images=10):
    """Display prediction results"""
    print(f"Displaying first {num_images} results...\n")
    
    result_images = glob.glob(str(Path(predict_dir) / "*.jpg"))[:num_images]
    
    if len(result_images) == 0:
        print("No result images found")
        return
    
    # Try to display with matplotlib if available
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        num_images = min(len(result_images), num_images)
        fig, axes = plt.subplots((num_images + 1) // 2, 2, figsize=(12, 4*((num_images + 1) // 2)))
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for idx, img_path in enumerate(result_images[:num_images]):
            img = mpimg.imread(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'Result {idx+1}')
        
        # Hide empty subplots
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(predict_dir).parent / 'results_summary.png', dpi=150, bbox_inches='tight')
        print(f"✓ Results summary saved to: {Path(predict_dir).parent / 'results_summary.png'}")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Results saved to:", predict_dir)
        for idx, img_path in enumerate(result_images, 1):
            print(f"  {idx}. {img_path}")

# ==============================================================================
# STEP 6: EXPORT MODEL
# ==============================================================================

def export_model(model_path, output_dir):
    """Package trained model for deployment"""
    print("\nPackaging model for deployment...")
    
    output_dir = Path(output_dir)
    model_dir = output_dir / "my_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model weights
    shutil.copy(model_path, model_dir / "my_model.pt")
    
    # Copy training results
    train_dir = Path(model_path).parent.parent
    if train_dir.exists():
        shutil.copytree(train_dir, model_dir / "train_results", dirs_exist_ok=True)
    
    # Create zip
    zip_path = output_dir / "my_model.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(model_dir.parent))
    
    print(f"✓ Model package created at: {zip_path}")
    print(f"✓ Model files at: {model_dir}")
    print("\nContents:")
    print(f"  - my_model.pt: Trained model weights")
    print(f"  - train_results/: Training metrics and results\n")
    
    return zip_path

# ==============================================================================
# STEP 7: CREATE INFERENCE SCRIPT
# ==============================================================================

def create_inference_script(output_dir):
    """Create a simple inference script"""
    
    script_content = '''#!/usr/bin/env python3
"""
Simple YOLO Inference Script
Run inference on images, videos, or webcam
"""

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLO Inference')
    parser.add_argument('--model', type=str, default='my_model.pt', help='Path to model')
    parser.add_argument('--source', type=str, default='0', help='Image/video/webcam (0 for webcam)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--show', action='store_true', help='Show results')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=args.show
    )
    
    print("Inference complete!")

if __name__ == '__main__':
    main()
'''
    
    script_path = Path(output_dir) / "my_model" / "inference.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✓ Inference script created at: {script_path}")
    print("\nUsage examples:")
    print("  python inference.py --model my_model.pt --source 0 --show  # Webcam")
    print("  python inference.py --model my_model.pt --source image.jpg --save  # Image")
    print("  python inference.py --model my_model.pt --source video.mp4 --save  # Video")
    print()

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Main training workflow"""
    
    print("\n" + "="*60)
    print("YOLO MODEL TRAINING - LOCAL/VSCODE VERSION")
    print("="*60 + "\n")
    
    # Step 0: Check GPU
    check_gpu()
    
    # Step 1: Setup workspace
    workspace = setup_workspace(WORKSPACE)
    
    # Step 2: Extract and prepare dataset
    custom_data_dir = workspace / "custom_data"
    if not extract_dataset(DATASET_ZIP, custom_data_dir):
        return
    
    # Step 3: Split dataset
    data_dir = workspace / "data"
    classes_path = custom_data_dir / CLASSES_FILE
    
    if not split_dataset(custom_data_dir, data_dir, train_ratio=0.9):
        return
    
    # Step 4: Create config file
    config_file = workspace / "data.yaml"
    if not create_config_file(classes_path, config_file, data_dir):
        return
    
    # Step 5: Train model
    try:
        train_model(
            config_file=str(config_file),
            model=MODEL,
            epochs=EPOCHS,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            project_dir=str(workspace / "runs")
        )
    except Exception as e:
        print(f"ERROR during training: {e}")
        return
    
    # Step 6: Test model
    best_model_path = workspace / "runs" / "train" / "weights" / "best.pt"
    val_images_dir = data_dir / "validation" / "images"
    
    test_model(
        model_path=str(best_model_path),
        source_dir=str(val_images_dir),
        project_dir=str(workspace / "runs")
    )
    
    # Display results
    predict_dir = workspace / "runs" / "predict"
    display_results(predict_dir, num_images=10)
    
    # Step 7: Export model
    export_model(best_model_path, workspace)
    
    # Step 8: Create inference script
    create_inference_script(workspace)
    
    print("\n" + "="*60)
    print("TRAINING WORKFLOW COMPLETE!")
    print("="*60)
    print(f"\nYour trained model is ready at:")
    print(f"  {workspace / 'my_model' / 'my_model.pt'}")
    print(f"\nModel package:")
    print(f"  {workspace / 'my_model.zip'}")
    print(f"\nTo run inference:")
    print(f"  cd {workspace / 'my_model'}")
    print(f"  python inference.py --model my_model.pt --source 0 --show")
    print("="*60 + "\n")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Check if ultralytics is installed
    try:
        import ultralytics
        import torch
    except ImportError as e:
        print("ERROR: Required packages not installed")
        print("\nPlease install dependencies:")
        print("  pip install ultralytics torch torchvision")
        print("\nFor GPU support (NVIDIA):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        exit(1)
    
    main()
```

---

## **FILE 2: requirements.txt**
```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
PyYAML>=6.0
matplotlib>=3.5.0
opencv-python>=4.8.0
pillow>=9.0.0
numpy>=1.21.0