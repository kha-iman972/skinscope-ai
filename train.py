import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def train():
    # 1. Set up data transforms (with augmentation)
    tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),                 # flip half the time
        transforms.RandomRotation(15),                     # rotate ±15°
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),# random crop + resize
        transforms.ColorJitter(0.1,0.1,0.1,0.1),           # slight color shifts
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],          # imagenet mean
                             [0.229,0.224,0.225])          # imagenet std
    ])

    # 2. Load dataset from data/ using ImageFolder
    data_dir = 'data'
    dataset = datasets.ImageFolder(data_dir, transform=tfms)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=2
    )

    # 3. Prepare the model (ResNet‑18) for fine‑tuning
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   # freeze all but final layer
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    # 4. Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 5. Set up loss and optimizer (only for final layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # 6. Training loop
    num_epochs = 5
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{num_epochs} — Loss: {epoch_loss:.4f}")

    # 7. Save the trained model and class mapping
    torch.save({
        'model_state': model.state_dict(),
        'classes': dataset.classes
    }, 'skin_model.pth')
    print("Training complete — model saved as skin_model.pth")

if __name__ == '__main__':
    train()
