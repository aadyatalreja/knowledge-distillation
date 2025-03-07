import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from models.vgg import VGG16Teacher
from models.mobilenet import MobileNetV2Student
from utils import get_training_dataloader, get_test_dataloader

# Hyperparameters
BATCH_SIZE = 64 # CHANGE TO 16 IR 32
EPOCHS = 200
LEARNING_RATE = 0.0005  # Lower learning rate
TEMPERATURE = 4.0  # Higher temperature helps softening logits
ALPHA = 0.5  # Give more weight to true labels
BETA = 0.3
INACTIVATION_PROB = 0.1  # Reduce neuron inactivation
EARLY_STOPPING_PATIENCE = 15

# Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA, beta=BETA):
    distill_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.LogSoftmax(dim=1)(student_logits / temperature),
        nn.Softmax(dim=1)(teacher_logits / temperature)
    ) * (temperature ** 2)

    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)

    entropy_loss = -torch.mean(torch.sum(nn.Softmax(dim=1)(student_logits) * nn.LogSoftmax(dim=1)(student_logits), dim=1))

    return alpha * distill_loss + (1 - alpha) * ce_loss + beta * entropy_loss

# Training Function with Early Stopping
def train(teacher, student, device, train_loader, optimizer):
    teacher.eval()
    student.train()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = teacher(data)

            student_output = student(data)
            loss = distillation_loss(student_output, teacher_output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch} - Loss: {epoch_loss:.4f}')

        # Early Stopping and Best Model Saving
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(student.state_dict(), "/content/best_model_vgg16.pth")  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

# Evaluation Function
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Main Script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_training_dataloader(BATCH_SIZE, std=0.5)
    test_loader = get_test_dataloader(BATCH_SIZE, std=0.5)

    # Load Pretrained Teacher Model
    teacher = VGG16Teacher(pretrained_path="/content/best_model_vgg16.pth").to(device)

    # Initialize Student Model
    student = MobileNetV2Student().to(device)

    # Train the Student Model
    print("Training Student Model with Knowledge Distillation...")
    optimizer_student = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    train(teacher, student, device, train_loader, optimizer_student)

    # Evaluate the Student Model
    student.load_state_dict(torch.load("/content/best_model_vgg16.pth"))  # Load best model
    evaluate(student, device, test_loader)

if __name__ == '__main__':
    main()