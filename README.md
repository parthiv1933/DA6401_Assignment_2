# CNN Image Classifier: From Scratch & Fine-tuning

A comprehensive implementation of CNN-based image classifiers using the iNaturalist dataset. This project is part of the DA6401 Assignment 2, focusing on two approaches:
1. Training a CNN from scratch with hyperparameter optimization
2. Fine-tuning a pre-trained GoogLeNet model

## Project Structure

```
.
├── README.md
├── train_A.py  # CNN implementation from scratch (Part A)
├── train_B.py  # Pre-trained model fine-tuning (Part B)
```

## Prerequisites

* Python 3.7+
* PyTorch
* torchvision
* wandb (Weights & Biases)
* CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision wandb
   ```

3. Set up wandb:
   ```bash
   wandb login
   ```

## Dataset

This project uses a subset of the iNaturalist dataset containing 10 classes of natural world images. The dataset should be organized as follows:

```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Usage

### Part A: Training CNN from Scratch

This implementation creates a custom CNN with 5 convolutional layers and allows experimentation with various hyperparameters including number of filters, activation functions, and regularization techniques.

```bash
python train_A.py --wandb_project 'DL_A2' --epochs 15 --batch_size 32 --learning_rate 0.0001 --activation relu --filters 32 --filter_scheme double --augment True --dropout 0.2 --batchnorm True --dense_units 256 --kernel_size 3 --img_dim 256 --train_dir './data/train' --test_dir './data/val'
```

#### Hyperparameters:
* `--activation`: Choose from 'selu', 'mish', 'leakyrelu', 'relu', 'gelu'
* `--filter_scheme`: Choose from 'same', 'half', 'double'
* `--augment`: Enable/disable data augmentation (True/False)
* `--batchnorm`: Enable/disable batch normalization (True/False)
* `--dropout`: Set dropout rate
* `--filters`: Base number of filters
* `--img_dim`: Input image dimension

### Part B: Fine-tuning Pre-trained Model

This implementation uses a pre-trained GoogLeNet model and fine-tunes it for the iNaturalist dataset with various freezing strategies.

```bash
python train_B.py --wandb_project 'DL_A2' --epochs 10 --batch_size 32 --strategy k_freeze --layers_to_freeze 15 --train_data_dir './data/train' --test_data_dir './data/val'
```

#### Fine-tuning Strategies:
* `--strategy`: Choose from:
  * `all_freeze`: Freeze all layers except the final fully connected layer
  * `k_freeze`: Freeze first k layers (controlled by `--layers_to_freeze`)
  * `no_freeze`: Don't freeze any layers (full fine-tuning)

## Features

### Part A: Custom CNN
* Configurable convolutional network architecture
* Multiple activation functions
* Adjustable filter schemes
* Data augmentation options
* Dropout and batch normalization for regularization
* Integration with wandb for experiment tracking
* Hyperparameter sweeps through wandb

### Part B: Pre-trained Model Fine-tuning
* GoogLeNet pre-trained on ImageNet
* Multiple layer freezing strategies
* Automatic handling of input dimensions and output layer adaptation
* Comprehensive data augmentation
* Performance visualization with wandb

## Experiment Tracking

Both implementations use Weights & Biases (wandb) for experiment tracking. Access experiment results by:
1. Creating an account on wandb.ai
2. Running the code with your wandb account
3. Viewing the results on the wandb dashboard

## Results

The project includes implementations for training and evaluating CNN models on the iNaturalist dataset. The performance metrics depend on hyperparameter choices and training duration. Refer to the wandb dashboard for detailed experiment results.

