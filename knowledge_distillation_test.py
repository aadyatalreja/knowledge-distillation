import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import time
from ptflops import get_model_complexity_info
from torchsummary import summary

# CIFAR-10 dataset mean and std
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2470, 0.2435, 0.2616)


def get_network(net_name):
    """Return the appropriate network based on name"""
    if net_name == 'vgg16':
        from torchvision.models import vgg16_bn
        model = vgg16_bn(pretrained=False)
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 10)
        )
    elif net_name == 'mobilenet':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(1280, 10)
    else:
        raise ValueError(f"Unknown network: {net_name}")
    return model


def get_test_dataloader(mean, std, batch_size=16, num_workers=4, shuffle=False):
    """Return test dataloader for CIFAR-10"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar10_test = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True,
        transform=transform_test
    )
    
    cifar10_test_loader = DataLoader(
        cifar10_test,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size
    )
    
    return cifar10_test_loader


if __name__ == '__main__':
    torch.manual_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mobilenet', help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    # Build network
    net = get_network(args.net)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    print(net)
    net.to(device)
    net.eval()
    
    # Get test data loader
    cifar10_test_loader = get_test_dataloader(
        CIFAR10_TRAIN_MEAN,
        CIFAR10_TRAIN_STD,
        batch_size=args.b,
        num_workers=4
    )
    
    # Calculate model complexity metrics
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, verbose=True)
    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params}")
    
    # Print model summary
    summary(net, (3, 32, 32), device=device.type)
    
    # Warm-up
    for image, label in cifar10_test_loader:
        image = image.to(device)
        label = label.to(device)
        output = net(image)
        break
    
    # Inference time measurement and accuracy calculation
    start_time = time.time()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar10_test_loader):
            print("iteration: {} \t total {} iterations".format(n_iter + 1, len(cifar10_test_loader)))
            
            image = image.to(device)
            label = label.to(device)
            
            output = net(image)
            
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            
            correct_5 += correct[:, :5].sum()
            correct_1 += correct[:, :1].sum()
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    print("\nTest Results:")
    print("=" * 50)
    print("Top-1 accuracy: {:.4f}".format(correct_1 / len(cifar10_test_loader.dataset)))
    print("Top-5 accuracy: {:.4f}".format(correct_5 / len(cifar10_test_loader.dataset)))
    print("Top-1 error: {:.4f}".format(1 - correct_1 / len(cifar10_test_loader.dataset)))
    print("Top-5 error: {:.4f}".format(1 - correct_5 / len(cifar10_test_loader.dataset)))
    print("=" * 50)
    print(f"Parameter count: {sum(p.numel() for p in net.parameters())}")
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Samples per second: {len(cifar10_test_loader.dataset) / inference_time:.2f}")
    time_per_step_ms = (inference_time / len(cifar10_test_loader)) * 1000
    print(f"Time per batch: {time_per_step_ms:.4f} milliseconds")
