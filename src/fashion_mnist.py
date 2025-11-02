import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FashionClassifier
# преобразования (можно добавить аугментации)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform = transforms.Compose([
    transforms.ToTensor(),  # превращаем в тензор
    transforms.Normalize((0.1307,), (0.3081,))  # нормализация [-1, 1]
])

# загрузка
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# dataloaders

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = FashionClassifier()

model.train()

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001) 

for epoch in range(15):
    for images, targets in train_loader:
        optimizer.zero_grad()
        predict = model(images)
        loss = loss_func(predict,targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = loss_func(outputs, labels)
        test_loss += loss.item()

        # считаем количество правильных предсказаний
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

# усредняем loss
test_loss /= len(test_loader)

# считаем accuracy
accuracy = 100. * correct / len(test_dataset)

print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "fashion_mnist_model.pth")
# использование
# model = FashionClassifier()               # создаём такую же архитектуру
# model.load_state_dict(torch.load("fashion_mnist_model.pth"))
# model.eval()   # переводим в режим инференса

