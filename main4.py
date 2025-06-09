import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import CLIPProcessor, CLIPModel


def main():
    # 1. Загрузка модели и процессора
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 2. Загрузка изображения
    image_path = r"C:\Users\user\Desktop\кошка.jpg"  # ← RAW string для Windows
    image = Image.open(image_path).convert("RGB")  # ← обязательно в RGB

    # 3. Текстовые описания
    text_descriptions = ["a photo of a cat", "a photo of a dog"]

    # 4. Обработка входных данных
    inputs = processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)

    # 5. Инференс без градиентов
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. Получаем вероятности
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).numpy().flatten()

    # 7. Визуализация: изображение + график вероятностей
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Изображение
    image_array = np.array(image)
    axes[0].imshow(image_array)
    axes[0].axis("off")
    axes[0].set_title("Исходное изображение")

    # График вероятностей
    sns.barplot(x=probs, y=text_descriptions, ax=axes[1], palette="Blues_d")
    axes[1].set_title("Вероятность соответсвия описанию")
    axes[1].set_xlabel("Вероятность")
    axes[1].set_ylabel("Описание")

    # Подписи над столбцами
    for index, value in enumerate(probs):
        axes[1].text(value, index, f"{value:.4f}", color='black', va='center')

    plt.tight_layout()
    plt.show()

    # 8. Вывод в консоль
    print("Модель ответила:")
    for i, t in enumerate(text_descriptions):
        print(f"{t}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()