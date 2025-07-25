import torch
from torchvision import transforms, models
from PIL import Image

# 1. Load checkpoint
checkpoint = torch.load('skin_model.pth', map_location='cpu')
classes   = checkpoint['classes']

# 2. Recreate the network architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 3. Preprocessing transforms (must match train.py)
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def analyze_skin(image_path):
    # Load and prepare image
    img = Image.open(image_path).convert('RGB')
    tensor = tfms(img).unsqueeze(0)  # add batch dim
    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        _, idx = outputs.max(1)
    return classes[idx]
