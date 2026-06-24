# Cats and Rabbit Image Classifier using ResNet18 and EfficientNet-B0

This project focuses on binary image classification to distinguish between cats and rabbits using deep learning. Two convolutional neural network architectures were studied and compared: **ResNet18** and **EfficientNet-B0**. The project includes both transfer learning using pretrained torchvision models and manual from-scratch implementations for deeper architectural understanding.

## Project Overview

The main objective of this project is to build an accurate image classification system for cat and rabbit images and compare the performance of two modern CNN architectures. The experiments evaluate model accuracy, precision, recall, loss behavior, convergence speed, and generalization quality.

## Models Used

- **ResNet18**: A residual network that uses skip connections to improve gradient flow and training stability.
- **EfficientNet-B0**: A lightweight and efficient CNN that uses compound scaling and MBConv blocks for strong performance with fewer parameters.

The repository contains:
- **Transfer learning models** using pretrained ImageNet weights.
- **Manual implementations** of ResNet18 and EfficientNet-B0 trained from scratch.

## Dataset

The dataset consists of cat and rabbit images collected from Kaggle. Images were preprocessed using:
- Resizing to 224 × 224
- Normalization using ImageNet statistics
- Data augmentation such as random horizontal flip and random rotation

The dataset was split into training and validation sets using an 80/20 ratio.

## Project Structure

```bash
.
├── dataset/
├── script-direct.py
├── script-manual.py
├── requirements.txt
├── readme.md
├── project report.pdf
└── result images / plots
```

## Requirements

This project requires Python 3.9.

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python3.9 -m venv .venv
   ```

2. Activate the environment:
   - Linux/MacOS:
     ```bash
     source .venv/bin/activate
     ```
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

- Run transfer learning models:
  ```bash
  python script-direct.py
  ```

- Run manual from-scratch models:
  ```bash
  python script-manual.py
  ```

## Training Details

- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 15
- Loss function: CrossEntropyLoss
- Input image size: 224 × 224

## Results Summary

The project showed that EfficientNet-B0 outperformed ResNet18 in both manual and transfer learning settings. Transfer learning significantly improved performance for both architectures. EfficientNet-B0 achieved better final accuracy, faster convergence, fewer misclassifications, and stronger generalization compared to ResNet18.


## Notes

`script-direct.py` uses torchvision pretrained ResNet18 and EfficientNet-B0 models for transfer learning.  
`script-manual.py` contains manual implementations of ResNet18 and EfficientNet-B0 trained from scratch.
