import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

def get_dataloaders(batch_size=128, seed=42, root='./data'):
    
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=2, worker_init_fn=worker_init_fn, pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return trainloader, testloader, train_set, test_set, classes

def get_trak_loader(train_set, batch_size=128, seed=42):
    # Non-shuffled loader for TRAK attribution
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        
    return torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False,
        num_workers=2, worker_init_fn=worker_init_fn
    )