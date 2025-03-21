import os
import sys
import argparse
import time
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0005
TEMPERATURE = 4.0
ALPHA = 0.5  # Weight for distillation loss
BETA = 0.3   # Weight for entropy loss
INACTIVATION_PROB = 0.1
EARLY_STOPPING_PATIENCE = 15
SAVE_EPOCH = 10  # Save model every X epochs

# CIFAR-10 dataset mean and std
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2470, 0.2435, 0.2616)


def get_network(net_name):
    """Return the appropriate network based on args"""
    if net_name == 'vgg16':
        from torchvision.models import vgg16_bn
        model = vgg16_bn(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )
    elif net_name == 'mobilenet':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(pretrained=False)
        model.classifier[-1] = nn.Linear(1280, 10)
    else:
        raise ValueError(f"Unknown network: {net_name}")
    return model


def get_dataloader(train=True, batch_size=128, num_workers=2, shuffle=True, pin_memory=True):
    """Return training or test dataloader for CIFAR-10"""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ]) if train else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)


def inactivate_neurons(x, prob=INACTIVATION_PROB):
    """Randomly inactivate neurons in the output tensor"""
    mask = (torch.rand_like(x, device=x.device) > prob).float()
    return x * mask


def distillation_loss(student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA, beta=BETA):
    """Knowledge distillation loss with entropy regularization"""
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)

    distill_loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)

    probs = F.softmax(student_logits, dim=1)
    log_probs = F.log_softmax(student_logits, dim=1)
    entropy_loss = -torch.mean(torch.sum(probs * log_probs, dim=1))

    return alpha * distill_loss + (1 - alpha) * ce_loss + beta * entropy_loss


def train(teacher, student, epoch, train_loader, optimizer, device):
    """Training function for one epoch with neuron inactivation"""
    start = time.time()
    print(f"Training epoch: {epoch}")

    teacher.eval()
    student.train()

    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = inactivate_neurons(teacher(inputs))

        student_logits = student(inputs)
        loss = distillation_loss(student_logits, teacher_logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(train_loader)
    print(f'Training Epoch {epoch} - Avg Loss: {avg_loss:.4f} - Time: {time.time() - start:.2f}s')

    return avg_loss


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f'Validation Accuracy: {acc:.4f}')
    return acc


def main():
    class Args:
        gpu = True
        b = BATCH_SIZE
        lr = LEARNING_RATE
        teacher = 'best_model_vgg16.pth'
        student = 'best_model_mobilenetv2.pth'
        output_dir = './checkpoint'

    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    train_loader = get_dataloader(train=True, batch_size=args.b, shuffle=True, pin_memory=device.type == 'cuda')
    test_loader = get_dataloader(train=False, batch_size=args.b, shuffle=False, pin_memory=device.type == 'cuda')

    teacher = get_network('vgg16').to(device)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device), strict=False)
    teacher.eval()

    student = get_network('mobilenet').to(device)
    student.load_state_dict(torch.load(args.student, map_location=device), strict=False)
    student.train()

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # Create checkpoint directory
    os.makedirs(args.output_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(teacher, student, epoch, train_loader, optimizer, device)
        acc = evaluate(student, test_loader, device)

        # Save the best model
        if acc > best_acc:
            best_acc = acc
            best_model_path = os.path.join(args.output_dir, f'kd_mobilenet-{epoch}-best.pth')
            print(f'Saving best model to {best_model_path}')
            torch.save(student.state_dict(), best_model_path)

        # Save model periodically
        if epoch % SAVE_EPOCH == 0:
            checkpoint_path = os.path.join(args.output_dir, f'kd_mobilenet-{epoch}-regular.pth')
            print(f'Saving checkpoint to {checkpoint_path}')
            torch.save(student.state_dict(), checkpoint_path)

    print(f"Training completed. Best accuracy: {best_acc*100:.2f}%")


if _name_ == '_main_':
    main()
