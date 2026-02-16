# YOLO Training for Local/VSCode Environment

Train YOLO11, YOLOv8, or YOLOv5 object detection models on your local machine with VSCode.

## Prerequisites

- **Python 3.8-3.11**
- **NVIDIA GPU** (optional but highly recommended)
  - CUDA Toolkit installed
  - cuDNN installed
- **8GB+ RAM** (16GB recommended)
- **Dataset** in YOLO format

## Quick Start

### 1. Clone or Download

Download all files to your project folder:
```
your_project/
├── train_yolo_local.py
├── requirements.txt
├── README.md
├── data.zip  (your dataset)
```

### 2. Create Virtual Environment
```bash
# Create environment
python -m venv yolo_env

# Activate environment
# Windows:
yolo_env\Scripts\activate
# Linux/Mac:
source yolo_env/bin/activate
```

### 3. Install Dependencies
```bash
# For CPU only:
pip install -r requirements.txt

# For GPU (NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Prepare Dataset

Create `data.zip` with this structure:
```
data.zip/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── classes.txt
```

**classes.txt format** (one class per line):
```
class1
class2
class3
```

**Label format** (.txt files):
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalized to 0-1.

### 5. Configure Training

Edit `train_yolo_local.py` configuration section:
```python
DATASET_ZIP = "data.zip"      # Your dataset path
MODEL = "yolo11s.pt"           # Model size (n/s/m/l/xl)
EPOCHS = 60                    # Training epochs
IMG_SIZE = 640                 # Image resolution
BATCH_SIZE = 16                # Batch size (reduce if OOM error)
DEVICE = 0                     # 0 for GPU, 'cpu' for CPU
```

### 6. Run Training
```bash
python train_yolo_local.py
```

The script will:
1. Check GPU availability
2. Extract and split dataset (90% train, 10% validation)
3. Create configuration files
4. Train the model
5. Test on validation set
6. Package model for deployment

## Training Output

After training completes, you'll find:
```
yolo_training/
├── data/                      # Processed dataset
├── runs/
│   ├── train/
│   │   ├── weights/
│   │   │   ├── best.pt       # Best model weights
│   │   │   └── last.pt       # Last epoch weights
│   │   ├── results.png       # Training metrics
│   │   └── confusion_matrix.png
│   └── predict/               # Validation predictions
└── my_model/
    ├── my_model.pt            # Trained model
    ├── inference.py           # Inference script
    ├── train_results/         # Training logs
    └── my_model.zip           # Complete package
```

## Running Inference

### Option 1: Using the generated script
```bash
cd yolo_training/my_model

# Webcam
python inference.py --model my_model.pt --source 0 --show

# Image
python inference.py --model my_model.pt --source test.jpg --save

# Video
python inference.py --model my_model.pt --source video.mp4 --save

# Folder of images
python inference.py --model my_model.pt --source /path/to/images/ --save
```

### Option 2: Using Python directly
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo_training/my_model/my_model.pt')

# Run inference
results = model.predict(
    source='image.jpg',  # or '0' for webcam, 'video.mp4', etc.
    conf=0.25,           # Confidence threshold
    save=True,           # Save results
    show=True            # Display results
)

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        print(f"Class: {int(box.cls)}, Confidence: {float(box.conf):.2f}")
```

## Configuration Options

### Model Sizes

| Model | Speed | Accuracy | Parameters | Best Use Case |
|-------|-------|----------|------------|---------------|
| yolo11n.pt | Fastest | Lowest | 2.6M | Edge devices, real-time |
| yolo11s.pt | Fast | Good | 9.4M | **Recommended** balanced option |
| yolo11m.pt | Medium | Better | 20.1M | Higher accuracy needed |
| yolo11l.pt | Slow | High | 25.3M | Maximum accuracy |
| yolo11xl.pt | Slowest | Highest | 56.9M | Research, offline processing |

### Training Parameters

**EPOCHS**: Number of training iterations
- Small dataset (<200 images): 60-100
- Medium dataset (200-500): 40-80
- Large dataset (>500): 30-60

**BATCH_SIZE**: Images per training step
- High-end GPU (24GB+): 32-64
- Mid-range GPU (8-12GB): 16-32
- Low-end GPU (4-6GB): 8-16
- CPU: 4-8

**IMG_SIZE**: Input resolution
- Fast inference: 480
- Standard: 640 (recommended)
- High accuracy: 1280

**DEVICE**: Computing device
- `0`: Use first GPU
- `1`: Use second GPU
- `'cpu'`: Use CPU only
- `[0, 1]`: Use multiple GPUs

## Troubleshooting

### Out of Memory (OOM) Error
```bash
# Reduce batch size
BATCH_SIZE = 8  # or 4

# Use smaller model
MODEL = "yolo11n.pt"

# Reduce image size
IMG_SIZE = 480
```

### No GPU Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Low Training Accuracy

1. **Check dataset quality**
   - Verify label accuracy
   - Ensure diverse images
   - Balance class distribution

2. **Increase training**
   - More epochs (100-200)
   - Larger model (yolo11m or yolo11l)

3. **Add more data**
   - Aim for 500+ images per class
   - Use data augmentation

### Training Too Slow

1. **Enable GPU** if available
2. **Reduce parameters**:
   - Smaller model (yolo11n)
   - Lower resolution (480)
   - Fewer epochs

### Dataset Errors
```bash
# Check folder structure
data.zip/
├── images/    # Must contain image files
├── labels/    # Must contain .txt label files
└── classes.txt  # Must exist with class names
```

## Advanced Usage

### Resume Training

If training is interrupted, resume from last checkpoint:
```python
from ultralytics import YOLO

model = YOLO('yolo_training/runs/train/weights/last.pt')
model.train(resume=True)
```

### Export to Different Formats
```python
from ultralytics import YOLO

model = YOLO('yolo_training/my_model/my_model.pt')

# TensorFlow Lite (mobile/edge)
model.export(format='tflite')

# ONNX (cross-platform)
model.export(format='onnx')

# TensorRT (NVIDIA GPUs)
model.export(format='engine')

# CoreML (iOS)
model.export(format='coreml')
```

### Custom Training Parameters

Edit the training call in `train_yolo_local.py`:
```python
results = model_obj.train(
    data=config_file,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device,
    project=project_dir,
    name='train',
    # Additional parameters:
    patience=50,        # Early stopping patience
    save_period=10,     # Save checkpoint every N epochs
    workers=8,          # Data loader workers
    optimizer='AdamW',  # Optimizer choice
    lr0=0.01,          # Initial learning rate
    augment=True,      # Enable augmentation
    mosaic=1.0,        # Mosaic augmentation
    mixup=0.0,         # Mixup augmentation
)
```

## Performance Benchmarks

Approximate training times for 200 images, 60 epochs:

| Hardware | Model | Time |
|----------|-------|------|
| RTX 4090 | yolo11s | 5-10 min |
| RTX 3080 | yolo11s | 10-15 min |
| RTX 3060 | yolo11s | 15-25 min |
| GTX 1660 | yolo11s | 30-45 min |
| CPU (i7) | yolo11s | 4-6 hours |

## Resources

- [Ultralytics Documentation](https://docs.ultralytics.com)
- [YOLO Official Repo](https://github.com/ultralytics/ultralytics)
- [Dataset Labeling Tools](https://labelstud.io)
- [Pre-trained Models](https://github.com/ultralytics/assets/releases)

## Support

For issues or questions:
1. Check this README
2. Review error messages carefully
3. Search [Ultralytics Issues](https://github.com/ultralytics/ultralytics/issues)
4. Check [Ultralytics Docs](https://docs.ultralytics.com)

## License

This training script is provided as-is for educational purposes.
YOLO models are subject to AGPL-3.0 license.
```

---

## **FILE 4: .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
yolo_env/
ENV/
env/

# Training outputs
yolo_training/
runs/
*.pt
*.pth
*.onnx
*.engine
*.tflite

# Dataset
data.zip
dataset/
custom_data/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
wandb/

# Temporary
*.tmp
temp/