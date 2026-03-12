# src/__init__.py

# 1. Expose the Model architectures
from .model import ResNet18, ResNet50, ResNet101

# 2. Expose the Data tools
from .data import get_dataloaders, get_trak_loader

# 3. Expose the Training engine
from .trainer import train_model

# 4. Expose the Analysis tools
from .analysis import run_trak_analysis

# 5. Expose Utilities
from .utils import set_seed, get_device

# This list defines what happens if someone types: "from src import *"
__all__ = [
    'ResNet18', 
    'ResNet50', 
    'ResNet101',
    'get_dataloaders',
    'get_trak_loader',
    'train_model',
    'run_trak_analysis',
    'set_seed',
    'get_device'
]