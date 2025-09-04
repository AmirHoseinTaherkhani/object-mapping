# Object Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade object detection system built with YOLOv8, designed for production deployment with MLOps best practices.

## 🚀 Features

- **State-of-the-art Detection**: YOLOv8-based object detection
- **Production Ready**: Docker containers, API endpoints, monitoring
- **MLOps Integration**: DVC for data versioning, automated CI/CD
- **Scalable Architecture**: Modular design for easy extension
- **Comprehensive Testing**: Unit and integration tests
- **API First**: RESTful API for seamless integration

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git with DVC
- AWS CLI (for model storage)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ObjectMapping.git
   cd ObjectMapping
   ```

2. **Set up environment**
   ```bash
   conda create -n object-detection python=3.9
   conda activate object-detection
   pip install -r requirements/base.txt
   ```

3. **Download models**
   ```bash
   dvc pull
   ```

4. **Run inference**
   ```bash
   python src/scripts/predict.py --input path/to/image.jpg
   ```

## 🛠 Installation

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ObjectMapping.git
cd ObjectMapping

# Create conda environment
conda env create -f environment.yml
conda activate object-detection

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements/dev.txt

# Set up pre-commit hooks
pre-commit install

# Download models and data
dvc pull
```

### Production Setup

```bash
# Using Docker
docker build -t object-detection .
docker run -p 8000:8000 object-detection

# Or using pip
pip install -r requirements/prod.txt
python src/api/main.py
```

## 📖 Usage

### Command Line Interface

#### Training
```bash
python src/scripts/train.py \
    --data configs/training/default.yaml \
    --model yolov8n \
    --epochs 100
```

#### Inference
```bash
# Single image
python src/scripts/predict.py --input image.jpg --output predictions/

# Batch processing
python src/scripts/predict.py --input images/ --output predictions/ --batch-size 16

# Video processing
python src/scripts/predict.py --input video.mp4 --output predictions/video_output.mp4
```

#### Evaluation
```bash
python src/scripts/evaluate.py \
    --model models/weights/yolov8n.pt \
    --data configs/data/test.yaml
```

### Python API

```python
from object_detection.inference import ObjectDetector

# Initialize detector
detector = ObjectDetector('models/weights/yolov8n.pt')

# Detect objects
results = detector.predict('path/to/image.jpg')

# Process results
for detection in results:
    print(f"Class: {detection.class_name}, Confidence: {detection.confidence}")
```

### REST API

Start the API server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Example request:
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

## 📚 API Documentation

Once the API is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc

### Endpoints

- `POST /detect` - Object detection on uploaded image
- `POST /batch-detect` - Batch object detection
- `GET /health` - Health check endpoint
- `GET /metrics` - Performance metrics

## 🔧 Development

### Project Structure

```
src/
├── object_detection/          # Main package
│   ├── models/               # Model definitions
│   ├── data/                 # Data handling
│   ├── training/             # Training logic
│   ├── inference/            # Inference pipeline
│   ├── utils/                # Utilities
│   └── api/                  # API endpoints
├── scripts/                  # CLI scripts
└── tests/                    # Test suite
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Models

1. Create model class in `src/object_detection/models/`
2. Add configuration in `configs/model/`
3. Update training script
4. Add tests

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t object-detection .

# Run container
docker run -p 8000:8000 object-detection

# Using docker-compose
docker-compose up
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -l app=object-detection
```

### AWS Deployment

```bash
# Deploy using Terraform
cd deployment/terraform
terraform init
terraform plan
terraform apply
```

## 📊 Model Performance

| Model | mAP@0.5 | Inference Time | Model Size |
|-------|---------|----------------|------------|
| YOLOv8n | 0.85 | 15ms | 6.2MB |
| YOLOv8s | 0.89 | 25ms | 21.5MB |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [DVC](https://dvc.org/) for data version control
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

## 📞 Support

- 📧 Email: a.h.taherkhani@gmail.com.com
---

**Made with ❤️ for the computer vision community**