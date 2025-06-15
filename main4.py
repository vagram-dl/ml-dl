import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import StandardScaler


class ImageClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def predict(self, image_path: str, text_descriptions: list) -> dict:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=text_descriptions,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1).numpy().flatten()
        return dict(zip(text_descriptions, probs))

    def visualize(self, image_path: str, text_descriptions: list):
        probs = self.predict(image_path, text_descriptions)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(Image.open(image_path))
        axes[0].axis("off")
        axes[0].set_title("Исходное изображение")

        # График вероятностей
        sns.barplot(
            x=list(probs.values()),
            y=list(probs.keys()),
            ax=axes[1],
            palette="Blues_d"
        )
        axes[1].set_title("Вероятности соответствия")
        plt.tight_layout()
        plt.show()


# ==================== Класс для анализа оттока клиентов ====================
class ChurnAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self._preprocess_data()

    def _preprocess_data(self):
        self.df.drop(columns=['customerID'], inplace=True)
        self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        self.df = pd.get_dummies(
            self.df,
            columns=['Contract', 'PaymentMethod'],
            drop_first=True
        )

        scaler = StandardScaler()
        num_cols = ['tenure', 'MonthlyCharges']
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])

    def analyze(self):
        print("Распределение классов Churn:\n", self.df['Churn'].value_counts(normalize=True))

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.histplot(data=self.df, x='tenure', hue='Churn', bins=30, kde=True)
        plt.title('Влияние срока обслуживания (tenure)')

        plt.subplot(1, 3, 2)
        sns.countplot(data=self.df, x='Contract_Month-to-month', hue='Churn')
        plt.title('Месячный контракт vs Отток')

        plt.subplot(1, 3, 3)
        sns.boxplot(data=self.df, x='Churn', y='MonthlyCharges')
        plt.title('Распределение платежей')

        plt.tight_layout()
        plt.show()


# ==================== Запуск обоих проектов ====================
if __name__ == "__main__":
    print("=== Классификация изображений ===")
    image_classifier = ImageClassifier()
    image_classifier.visualize(
        image_path="cat.jpg",  # Укажите путь к вашему изображению
        text_descriptions=["a photo of a cat", "a photo of a dog"]
    )

    print("\n=== Анализ оттока клиентов ===")
    churn_analyzer = ChurnAnalyzer(
        data_path="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )
    churn_analyzer.analyze()