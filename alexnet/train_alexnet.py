import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from torchvision.models import AlexNet_Weights
from torchvision import datasets
from torch.optim import SGD 
from torch.utils.data import DataLoader


def getDataset():
    # Default transforms for pretrained AlexNet 
    # (https://docs.pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html) 
    transform = AlexNet_Weights.IMAGENET1K_V1.transforms()

    # Load dataset path from ./params.yaml
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
        dataset_path = params['dataset_path']
    
    train = datasets.ImageFolder(
        root=f"{dataset_path}/train",
        transform=transform
    )
    valid = datasets.ImageFolder(
        root=f"{dataset_path}/valid",
        transform=transform
    )
    
    return train, valid


def alexnet_train(model, train_dataset, valid_dataset, num_epochs=50, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005) # Stochastic Gradient Descent optimizer

    best_valid_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}", flush=True)

        # Training phase
        model.train()
        running_loss = 0.0

        for train_inputs, train_labels in train_loader:
            inputs, labels = train_inputs.to(device), train_labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valid_loader:
                inputs, labels = val_inputs.to(device), val_labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_dataset)

        print(f"Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}", flush=True)

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'alexnet.pth')
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs", flush=True)

            if epochs_no_improve >= patience:
                print("Early stopping triggered", flush=True)
                break
    
    model.load_state_dict(torch.load('alexnet.pth'))
    print("\n\nTraining complete. Best model saved as 'alexnet.pth'", flush=True)
    return model


if __name__ == "__main__":

    print("Loading dataset...", flush=True)
    train, valid = getDataset()

    print("Creating AlexNet model...", flush=True)
    alexnet = models.alexnet(weights='IMAGENET1K_V1')
    num_classes = len(train.classes)
    alexnet.classifier[6] = nn.Linear(4096, num_classes)

    print("Training AlexNet model...", flush=True)
    alexnet = alexnet_train(alexnet, train, valid)