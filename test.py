import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vgg import VGG  # Adjust this import based on your project structure
from torchsummary import summary

# Ensure the device is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    # Define transforms for the dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-100 dataset
    testset = datasets.cifar10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return testloader

def log_results(correct_1, correct_5, total, inference_time, testloader, net, device):
    top1_error = 1 - (correct_1 / total)
    top5_error = 1 - (correct_5 / total)
    time_per_inference = inference_time / len(testloader)
    
    print(f"Top 1 error: {top1_error:.4f}")
    print(f"Top 5 error: {top5_error:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Time per inference step: {time_per_inference * 1000:.4f} ms")
    
    # Ensure the model is on the correct device
    net = net.to(device)
    summary(net, (3, 32, 32), device=str(device))

def main():
    # Load test data
    cifar10_test_loader = load_data()

    # Initialize the model (adjust this to match your project's model initialization)
    net = VGG('VGG16')
    
    # Load model weights
    net.load_state_dict(torch.load("/content/vgg16-197-best.pth", map_location=device))
    net = net.to(device)  # Move the model to the appropriate device
    net.eval()  # Set the model to evaluation mode
    
    # Initialize variables for tracking results
    correct_1 = 0
    correct_5 = 0
    total = 0
    inference_time = 0
    
    # Evaluate the model
    with torch.no_grad():
        for inputs, targets in cifar10_test_loader:
            # Move inputs and targets to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            outputs = net(inputs)
            end_time.record()
            torch.cuda.synchronize()  # Wait for GPU operations to finish
            
            inference_time += start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            
            # Calculate top-1 and top-5 accuracy
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top5 = outputs.topk(5, dim=1)
            
            correct_1 += (pred_top1.squeeze() == targets).sum().item()
            correct_5 += sum(targets[i] in pred_top5[i] for i in range(len(targets)))
            total += targets.size(0)
    
    # Log results
    log_results(correct_1, correct_5, total, inference_time, cifar10_test_loader, net, device)

if __name__ == "__main__":
    main()