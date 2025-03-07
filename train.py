import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vgg import VGG, make_layers, cfg

# Training function with accuracy calculation
def train(net, train_loader, optimizer, loss_function, device):
    start = time.time()
    net.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Testing function to evaluate model accuracy on CIFAR-10 test set
def test(net, test_loader, loss_function, device, best_acc, save_path):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    # Save best model based on test accuracy
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(net.state_dict(), save_path)
    
    return best_acc, accuracy

if _name_ == '_main_':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg16', help='network type (default: vgg16)')
    parser.add_argument('-b', type=int, default=128, help='batch size')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading with separate transforms for train/test
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=4)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=4)
    
    # Model selection
    if args.net == 'vgg16':
        net = VGG(make_layers(cfg['D'], batch_norm=True), num_class=10).to(device)
    else:
        raise ValueError(f'Unsupported network type: {args.net}')
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Learning Rate Scheduler: Decay LR at 50, 100, 150 epochs
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    best_acc = 0.0  # Track the best test accuracy
    save_path = f'./best_model_{args.net}.pth'
    
    for epoch in range(1, 201):  # Train for 200 epochs
        train_loss, train_acc = train(net, train_loader, optimizer, loss_function, device)
        best_acc, test_acc = test(net, test_loader, loss_function, device, best_acc, save_path)
        scheduler.step()  # Update learning rate
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    print(f'Training Completed! Best Test Accuracy: {best_acc:.2f}%')