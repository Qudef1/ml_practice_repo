import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# --- Загрузка модели ---
classes = np.load("clothing_classes.npy", allow_pickle=True)

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("fashion_resnet50_articleType.pth", map_location="cpu"))
model.eval()

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
            output = model(img_t)
            prob = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)
            label = classes[predicted.item()]
            conf = confidence.item() * 100

        await update.message.reply_text(
            f"👕 Это: *{label}*\n✅ Уверенность: {conf:.1f}%",
            parse_mode="Markdown"
        )
        os.remove(file_path)
    except Exception as e:
        await update.message.reply_text("❌ Ошибка. Попробуйте другое фото.")

# --- Запуск ---
if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # ← берётся из переменной окружения!
    if not TOKEN:
        raise ValueError("❌ Переменная окружения TELEGRAM_BOT_TOKEN не задана!")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("Бот запущен!")
    app.run_polling()