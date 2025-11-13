import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# --- Загрузка конфигурации ---
with open("num_classes.json", "r") as f:
    num_classes = json.load(f)

# --- Загрузка классов ---
classes = {
    "gender": np.load("classes_gender.npy", allow_pickle=True),
    "baseColour": np.load("classes_baseColour.npy", allow_pickle=True),
    "subCategory": np.load("classes_subCategory.npy", allow_pickle=True),
    "articleType": np.load("classes_articleType.npy", allow_pickle=True)
}

# --- Модель ---
from torchvision import models
import torch.nn as nn

class MultiTaskResNet50(nn.Module):
    def __init__(self, num_gender, num_colour, num_subcat, num_article):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features
        self.head_gender = nn.Linear(feature_dim, num_gender)
        self.head_colour = nn.Linear(feature_dim, num_colour)
        self.head_subCategory = nn.Linear(feature_dim, num_subcat)
        self.head_articleType = nn.Linear(feature_dim, num_article)

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        return (
            self.head_gender(features),
            self.head_colour(features),
            self.head_subCategory(features),
            self.head_articleType(features)
        )

# Создание и загрузка модели
model = MultiTaskResNet50(
    num_gender=num_classes["gender"],
    num_colour=num_classes["baseColour"],
    num_subcat=num_classes["subCategory"],
    num_article=num_classes["articleType"]
)
model.load_state_dict(torch.load("fashion_multitask_resnet50.pth", map_location="cpu"))
model.eval()

# Трансформации (должны совпадать с обучением!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Обработчик фото ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        file_path = "temp.jpg"
        await file.download_to_drive(file_path)

        img = Image.open(file_path).convert("RGB")
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            out_g, out_c, out_s, out_a = model(img_t)
            pred_g = classes["gender"][out_g.argmax().item()]
            pred_c = classes["baseColour"][out_c.argmax().item()]
            pred_s = classes["subCategory"][out_s.argmax().item()]
            pred_a = classes["articleType"][out_a.argmax().item()]

        response = (
            f"👕 *Тип:* {pred_a}\n"
            f"🏷️ *Категория:* {pred_s}\n"
            f"🎨 *Цвет:* {pred_c}\n"
            f"👤 *Пол:* {pred_g}"
        )
        await update.message.reply_text(response, parse_mode="Markdown")
        os.remove(file_path)

    except Exception as e:
        print(f"Ошибка: {e}")
        await update.message.reply_text("❌ Не удалось распознать одежду. Попробуйте другое фото.")

# --- Запуск ---
if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TOKEN = "7630775196:AAFMeW4q6s4ViKMtaAHP-zIdDdefwCsVkms"
    if not TOKEN:
        raise ValueError("Установите переменную окружения TELEGRAM_BOT_TOKEN")
    
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("✅ Бот запущен!")
    app.run_polling()