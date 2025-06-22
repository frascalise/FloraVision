import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import yaml


def getDataset():
    transform = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
    ])

    # Load dataset path from ./params.yaml
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
        dataset_path = params['dataset_path']
    
    test_dataset = datasets.ImageFolder(root=f"{dataset_path}", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader
    

if __name__ == "__main__":
    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    print("Loading dataset...", flush=True)
    test = getDataset()
    num_classes = len(test.dataset.classes)

    # Load the pre-trained AlexNet model
    print("Loading the AlexNet model...", flush=True)
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096,num_classes)  # Adjust the final layer for the number of classes
    model.load_state_dict(torch.load('alexnet.pth', map_location=device))  # Load the trained model (weights)
    model.to(device)
    model.eval()

    # Test the model
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    print("Testing the model...", flush=True)
    with torch.no_grad():
        counter = 0
        for inputs, labels in test:
            counter += 1
            print(f"Processing batch {counter}", flush=True)

            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, preds = torch.max(outputs, 1)

            # Update the total and correct counts (batch-wise)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # Move labels and preds to CPU for further processing
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.4f}')

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test.dataset.classes))
