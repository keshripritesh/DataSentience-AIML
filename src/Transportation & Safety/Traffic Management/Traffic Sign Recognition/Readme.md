# Traffic Sign Recognition System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

![Traffic Sign Recognition Demo](Images/Demo.jpg)

A deep learning-based system for recognizing and classifying traffic signs in images and real-time video streams. Built with Keras/TensorFlow, OpenCV, and Tkinter.

## Key Features
- 🚦 43-class traffic sign recognition
- 🖼️ Image classification through GUI
- 🎥 Real-time video processing capability
- 📊 Model training and evaluation scripts
- ✅ 95% test accuracy

## Dataset
The model was trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset containing:
- 50,000+ high-quality traffic sign images
- 43 distinct sign classes
- Various lighting and weather conditions

## System Architecture
```mermaid
graph LR
A[Input Image] --> B(Preprocessing)
B --> C[CNN Model]
C --> D[Prediction]
D --> E[Class Label]
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

> Note: A `requirements.txt` file is recommended. Here's what it should contain:
> ```text
> tensorflow==2.10.0
> keras==2.10.0
> pandas==1.5.0
> pillow==9.2.0
> scikit-learn==1.1.2
> matplotlib==3.6.0
> opencv-python==4.6.0.66
> ```

## Usage

### GUI Image Classification

Run the graphical interface:
```bash
python gui.py
```

1. Click "Upload an image"
2. Select a traffic sign image
3. Click "Classify Image"
4. View results in the GUI

### Real-time Video Processing

Run the video demonstration:
```bash
python future_improvements.py
```

### Model Training

To retrain the model:
```bash
python traffic_signs.py
```

## Project Structure
```text
traffic-sign-recognition/
├── utils.py                # Image preprocessing utilities
├── gui.py                  # Graphical user interface
├── traffic_signs.py        # Model training script
├── future_improvements.py  # Video processing demo
├── tests/                  # Unit tests
│   └── test_utils.py
├── model.h5                # Pretrained model (gitignored)
└── Images/                 # Screenshots and examples
    └── Demo.jpg            # GUI screenshot
```

## Results

The CNN model achieves:
- 95% test accuracy
- 15 training epochs
- Efficient inference time (<100ms/image)

## Customization

- To modify image preprocessing: Edit preprocess_image() in `utils.py`
- To change model architecture: Modify `traffic_signs.py`
- To add new sign classes: Update the classes dictionary in `gui.py`

## Contributing

Contributions are welcome! Please follow the steps in [CONTRIBUTING.md](../../../../Contributing.md)

## License

Distributed under the MIT License. See [`LICENSE`](../../../../License.md) for more information.