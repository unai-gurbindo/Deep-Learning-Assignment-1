###############################################################################################################################
# Importing the required libraries

# Data manipulation and processing
import pandas as pd  # Library for data manipulation and analysis
import json  # Library for handling JSON data
from sklearn.preprocessing import LabelEncoder  # Library for encoding categorical features into numerical values

# Machine learning and deep learning frameworks
import torchvision  # Library for computer vision tasks built on top of PyTorch
from torchvision.io import read_image  # Module for reading images using torchvision
from torchvision.transforms.v2 import (  # Module for image transformations
    Compose, Resize, Normalize, ConvertImageDtype,
    RandomHorizontalFlip, RandomResizedCrop, ColorJitter
)
import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network module in PyTorch
import torch.optim as optim  # Optimization algorithms in PyTorch
from torch.utils.data import DataLoader  # DataLoader for handling batches of data in PyTorch

# File system and array manipulation
import os  # Library for interacting with the operating system
import numpy as np  # Library for numerical computations in Python

# Input-output operations
import io  # Library for handling I/O operations in Python
import sys  # Library providing access to some variables used or maintained by the Python interpreter
from torchsummary import summary  # Module for summarizing PyTorch models
###############################################################################################################################


###############################################################################################################################
# Model architecture

config = {
  "run_name": "cnn_58",# Name of the run
  "model":
    { # Convolutional neural network architecture
      "n_conv_layers": 4,# Number of convolutional layers
      "out_channels": [32, 64, 128, 256],# Number of output channels in each convolutional layer
      "kernel_size": [3, 3, 3, 3], # Kernel size in each convolutional layer
      "stride": [1, 1, 1, 1], # Stride in each convolutional layer
      "padding": ["same", "same", "same", "same"], # Padding in each convolutional layer
      "activation": ["ReLU", "ReLU", "ReLU", "ReLU"], # Activation function in each convolutional layer
      "dropout_rate": [0.2, 0.2, 0.2, 0.2], # Dropout 2D rate in each convolutional layer
      "pool_kernel_size": [2, 2, 2, 2], # Kernel size for max pooling in each convolutional layer
      "num_flatten_features": 256, # Number of features after flattening the last convolutional layers 
      "use_batch_norm": True, # Whether to use batch normalization
      "use_adaptive_pooling": True, # Whether to use adaptive pooling with the last convolutional layer
      "fc_layers": { # Fully connected layers
        "n_fc_layers": 3, # Number of fully connected layers
        "activation": ["ReLU", "ReLU", "ReLU"], # Activation function in each fully connected layer
        "out_features": [256, 128, 64], # Number of output features in each fully connected layer
        "dropout_rate": [0.5, 0.3, 0.3] # Dropout rate in each fully connected layer
      },
      "output_features": 29 # Number of output classes
    },
  "training":
    {
      "batch_size": 128, # Batch size for training
      "epochs": 100, # Number of epochs (Mention that in this case the model was trained first for 300 epochs with a learning rate of 0.001 and then for 100 epochs with a learning rate of 0.0005)
      "optimizer": "Adam", # Optimizer for training
      "optimizer_params": {
        "lr": 0.0005,
      },
      "patience": 50, # Patience for early stopping
      "num_workers": 4 
    },
  "transformations": [
      {"name": "ConvertImageDtype"}, # Convert the image to PyTorch tensor
    ]
}

###############################################################################################################################
# Custom CNN Class Development

class CustomCNN(nn.Module):
    def __init__(self, model_config):
        super(CustomCNN, self).__init__()
        self.layers = nn.Sequential()  # Convolutional and pooling layers

        # Convolutional layer parameters
        n_conv_layers = model_config["n_conv_layers"]
        out_channels = model_config["out_channels"]
        kernel_size = model_config["kernel_size"]
        stride = model_config["stride"]
        padding = model_config["padding"]
        activation = model_config["activation"]
        dropout_rate = model_config["dropout_rate"]
        pool_kernel_size = model_config["pool_kernel_size"]

        in_channels = 3  # Assuming the input has 3 channels (e.g., RGB image)

        # Adding convolutional and pooling layers
        for i in range(n_conv_layers):
            # Convolutional layer
            self.layers.add_module(f"conv{i}", nn.Conv2d(in_channels, out_channels[i], kernel_size[i], stride[i], 0 if padding[i] == "valid" else 1))

            # Conditionally add batch normalization
            if model_config.get("use_batch_norm", False):
                self.layers.add_module(f"bn{i}", nn.BatchNorm2d(model_config["out_channels"][i]))
            
            # Activation
            if activation[i] in dir(nn):
                act_func = getattr(nn, activation[i])()
                self.layers.add_module(f"act{i}", act_func)
            
            # Pooling layer
            if pool_kernel_size[i]:
                self.layers.add_module(f"pool{i}", nn.MaxPool2d(kernel_size=pool_kernel_size[i]))
            
            # Dropout
            if dropout_rate[i] > 0:
                self.layers.add_module(f"dropout_2d{i}", nn.Dropout2d(dropout_rate[i]))

            in_channels = out_channels[i]

        # Optional adaptive pooling layer
        self.use_adaptive_pooling = model_config.get("use_adaptive_pooling", False)
        if self.use_adaptive_pooling:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc_layers = nn.Sequential()
        fc_config = model_config["fc_layers"]
        n_fc_layers = fc_config["n_fc_layers"]
        fc_out_features = fc_config["out_features"]
        fc_dropout_rate = fc_config["dropout_rate"]
        activation = fc_config["activation"]
        num_features = model_config["num_flatten_features"]  # This needs to be calculated based on your input size and the architecture of your CNN

        for i in range(n_fc_layers):
            self.fc_layers.add_module(f"fc{i}", nn.Linear(num_features, fc_out_features[i]))
            if activation[i] in dir(nn):
                act_func = getattr(nn, activation[i])()
                self.fc_layers.add_module(f"act{i}", act_func)
            if fc_dropout_rate[i] > 0:
                self.fc_layers.add_module(f"dropout_fc{i}", nn.Dropout(fc_dropout_rate[i]))
            num_features = fc_out_features[i]

        # Output layer
        self.output_layer = nn.Linear(num_features, model_config["output_features"])  # Assuming output_features corresponds to the number of classes

    def forward(self, x):
        x = self.layers(x)
        if self.use_adaptive_pooling:
            x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten the output for the FC layers
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x
###############################################################################################################################

# Model summary
model = CustomCNN(config['model'])

summary(model, input_size=(3, 256, 256))  # Assuming input size is 256x256x3