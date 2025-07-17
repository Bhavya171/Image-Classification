# Image Classification with Fine-Tuned ResNet50 on CIFAR-100

A deep learning project that implements transfer learning using a pre-trained ResNet50 model to classify images from the CIFAR-100 dataset, achieving competitive performance through fine-tuning techniques.

## Project Overview
This project demonstrates the power of transfer learning by adapting a ResNet50 model, originally trained on ImageNet, to classify images from the CIFAR-100 dataset. The model leverages pre-trained feature extraction capabilities while being fine-tuned for the specific task of classifying 100 different object categories.

## Features
- Transfer Learning: Utilizes pre-trained ResNet50 weights from ImageNet

- Custom Architecture: Adds custom classification layers for CIFAR-100 (100 classes)

- Data Preprocessing: Implements proper image resizing and normalization

- Fine-tuning Strategy: Selective layer freezing and unfreezing for optimal training

- Performance Monitoring: Tracks training and validation metrics

## Results
- Final Validation Accuracy: ~75%

- Training Epochs: 10

- Model Size: ~98.37 MB

- Total Parameters: 25,788,388 (25.7M trainable, 53K non-trainable)

## Model Architecture
- Component	Details
- Base Model	ResNet50 pre-trained on ImageNet
- Input Shape	(224, 224, 3)
- Global Pooling	GlobalAveragePooling2D
- Dense Layer	1024 units with ReLU activation
- Output Layer	100 units with Softmax activation
- Total Parameters	25,788,388
## Training Details
- Optimizer: Adam

- Loss Function: Sparse Categorical Crossentropy

- Batch Size: 32

- Epochs: 10

- Fine-tuning: Layers 143+ unfrozen

- Data Augmentation: ResNet50 preprocessing

## Technical Implementation
### Data Preprocessing
- Images resized from 32×32 to 224×224 pixels

- ResNet50-specific preprocessing applied

- Batch processing with TensorFlow datasets

### Transfer Learning Strategy
- Start with ImageNet pre-trained weights

- Replace final classification layer for 100 classes

- Freeze early feature extraction layers

- Fine-tune deeper layers for CIFAR-100 specific features

## Performance Metrics
| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|-------------------|---------------------|---------------|-----------------|
| 1     | 49.8%             | 69.1%               | 1.92          | 1.02            |
| 5     | 94.2%             | 72.8%               | 0.17          | 1.40            |
| 10    | 98.0%             | 74.9%               | 0.06          | 1.52            |
