import torch
from model import FashionClassifier
from torchvision import transforms
# тот же класс модели, что и при обучении
model = FashionClassifier()

# загрузка весов
model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location="cpu"))

# режим инференса
model.eval()

from PIL import Image

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

path = input("enter path of picture")
img = Image.open(path).convert("L")  # "L" = grayscale
img = transform(img).unsqueeze(0)  # (1, 1, 28, 28)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1).item()

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("Предсказание:", classes[pred])
print(output)