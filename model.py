import torch
from torchvision import transforms
from PIL import Image

# Load once on module import
checkpoint = torch.load('skin_model.pth', map_location='cpu')
classes = checkpoint['classes']
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Pre-processing transforms
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def analyze_skin(image_path):
    # 1. Load image
    img = Image.open(image_path).convert('RGB')
    # 2. Preprocess
    tensor = tfms(img).unsqueeze(0)  # add batch dimension
    # 3. Run model
    with torch.no_grad():
        outputs = model(tensor)
    # 4. Pick top prediction
    _, idx = outputs.max(1)
    return classes[idx]
