import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Data paths
DATA_DIR = "E:/IIT/FYP/My FYP/New Dataset"

# Image transformations with augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transformations
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Lightweight CNN model for mobile compatibility
class PetSkinDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PetSkinDiseaseModel, self).__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fifth convolutional block
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def train_model(animal_type, batch_size=32, num_epochs=25, learning_rate=0.001):
    # Data paths for specific animal type
    train_dir = os.path.join(DATA_DIR, "train", animal_type)
    valid_dir = os.path.join(DATA_DIR, "val", animal_type)
    test_dir = os.path.join(DATA_DIR, "test", animal_type)

    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=val_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=val_transforms)

    # Get class names and count
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"\n{animal_type.capitalize()} classes: {class_names}")

    # Handle class imbalance with weighted sampling
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = PetSkinDiseaseModel(num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # Training loop
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = 100 * valid_correct / valid_total

        # Update learning rate
        scheduler.step(valid_loss)

        # Print statistics
        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%')

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{animal_type}_model.pth')
            print(f'Model saved with validation loss: {valid_loss:.4f}')

    # Load best model for evaluation
    model.load_state_dict(torch.load(f'{animal_type}_model.pth'))

    # Test the model
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'\nTest Accuracy for {animal_type}: {test_acc:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print('Confusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))

    return model, class_names


if __name__ == "__main__":
    # Train models for both dogs and cats
    print("Training Dog Model...")
    dog_model, dog_classes = train_model("dog")

    print("\nTraining Cat Model...")
    cat_model, cat_classes = train_model("cat")

    # Save class names for inference
    torch.save({
        'dog_classes': dog_classes,
        'cat_classes': cat_classes
    }, 'class_names.pth')

    print("\nTraining complete. Models and class names saved.")
