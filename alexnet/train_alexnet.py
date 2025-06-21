from torchvision import transforms, datasets
import torchvision.models as models
import torch
from torch.optim import SGD 
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml


def getDataset():
    transform = transforms.Compose([
        transforms.Resize(227), # Resize to 227x227 for AlexNet (standard AlexNet's input size)
        transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean and std
    ])

    # Load dataset path from ./params.yaml
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
        dataset_path = params['dataset_path']
    
    # ImageFolder expects a directory structure like:
    # dataset_path/train/class_name/xxx.png
    train = datasets.ImageFolder(
        root=f"{dataset_path}/train",
        transform=transform
    )
    valid = datasets.ImageFolder(
        root=f"{dataset_path}/valid",
        transform=transform
    )
    
    return train, valid


def alexnet_train(model, train_dataset, valid_dataset, num_epochs=50, patience=5):
    # Set the device to GPU if available, otherwise CPU and move the model to that device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataLoader divides the dataset into batches for training and validation
    # It's efficient because it doesn't load the entire dataset into memory at once
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005) # Stochastic Gradient Descent optimizer

    # Early stopping parameters
    best_valid_loss = float('inf') # Initialize best validation loss to infinity so that any valid loss will be lower
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}", flush=True)

        # Model set to training mode (dropout enabled -> some neurons will be randomly dropped)
        model.train()
        running_loss = 0.0

        for train_inputs, train_labels in train_loader:
            # Move input batch and label batch to the device
            inputs, labels = train_inputs.to(device), train_labels.to(device)

            optimizer.zero_grad()   # Set gradients to zero (prevent accumulation from previous iterations)
            outputs = model(inputs)
            loss = criterion(outputs, labels)   # Calculate loss (in this case, CrossEntropyLoss)

            loss.backward()     # Computes the loss gradient with respect to the model parameters 
                                # (it fill the .grad attributes of the model parameters)

            optimizer.step()    # Applies the gradients (calulated in loss.backward()) to the model parameters

            # Accumulate loss for the batch
            # inputs.size(0) gives the batch size, so we multiply batch avg. loss by batch size to get total loss for the batch
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        
        # Model set to evaluation mode (dropout disabled -> all neurons are used)
        model.eval()
        valid_loss = 0.0

        # Disable gradient calculation for validation to save memory and computation
        with torch.no_grad():
            for val_inputs, val_labels in valid_loader:
                inputs, labels = val_inputs.to(device), val_labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        valid_loss /= len(valid_dataset)

        print(f"Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}", flush=True)

        # Early stopping
        # If validation loss improves (for example, from 0.18 to 0.17), save the model
        # If validation loss does not improve for 'patience' epochs, stop training
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
    
    print("\n\nTraining complete. Best model saved as 'alexnet.pth'", flush=True)


if __name__ == "__main__":

    print("Loading dataset...", flush=True)
    train, valid = getDataset()

    print("Creating AlexNet model...", flush=True)
    alexnet = models.alexnet(pretrained=False)
    num_classes = len(train.classes)
    alexnet.classifier[6] = nn.Linear(4096, num_classes)

    print("Training AlexNet model...", flush=True)
    alexnet_train(alexnet, train, valid)