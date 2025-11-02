import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision import models


classes = np.load("label_encoder_classes.npy", allow_pickle=True)

print(*classes)

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("fashion_resnet50.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("download.jpg").convert("RGB")
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_t)
    pred = output.argmax(dim=1).item()
    print("Предсказание:", classes[pred])