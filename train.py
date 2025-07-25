import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

def train():
    # 1. Transform: resize to 224×224 and normalize
    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # 2. Load dataset from data/ folder
    data_dir = 'data'
    dataset = datasets.ImageFolder(data_dir, transform=tfms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. Fine-tune a pretrained model (e.g., ResNet18)
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # freeze base layers
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    # 4. Train loop (one epoch for demonstration)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # 5. Save trained weights
    torch.save({
        'model_state': model.state_dict(),
        'classes': dataset.classes
    }, 'skin_model.pth')
    print("Training complete, model saved as skin_model.pth")

if __name__ == '__main__':
    train()
