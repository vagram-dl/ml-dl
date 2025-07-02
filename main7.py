import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pydicom
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class MultimodalCancerDataset(Dataset):
    def __init__(self, image_paths, tabular_data, labels, img_size=256, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

        # Проверка существования файлов
        self._check_paths_exist()

        # Нормализация табличных данных
        self.scaler = StandardScaler()
        self.tabular_data = self.scaler.fit_transform(tabular_data)

        # Аугментации
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]) if augment else None

    def _check_paths_exist(self):
        for path in self.image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"DICOM file not found: {path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            dicom = pydicom.dcmread(self.image_paths[idx])
            image = dicom.pixel_array.astype(np.float32)

            image = (image - np.percentile(image, 1)) / (np.percentile(image, 99) - np.percentile(image, 1) + 1e-8)
            image = np.clip(image, 0, 1)

            image = torch.from_numpy(image).unsqueeze(0)
            image = F.interpolate(image.unsqueeze(0), size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False).squeeze(0)

            if self.augment and self.transform:
                image = self.transform(image)

            return {
                'image': image,
                'tabular': torch.FloatTensor(self.tabular_data[idx]),
                'label': torch.FloatTensor([self.labels[idx]])
            }

        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {str(e)}")
            return {
                'image': torch.zeros((1, self.img_size, self.img_size)),
                'tabular': torch.zeros(self.tabular_data.shape[1]),
                'label': torch.FloatTensor([0])
            }


class AttentionBlock(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.ReLU(),
            nn.Linear(features // 2, features),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights


class MultimodalCancerModel(nn.Module):
    def __init__(self, tabular_size):
        super().__init__()

        # Ветвь для изображений
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )

        # Ветвь для табличных данных
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_size, 64),
            nn.BatchNorm1d(64),  # Исправлено с BatchNorm2d
            nn.ReLU()
        )

        # Механизм внимания
        self.attention = AttentionBlock(128)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            self.attention,
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Обработка изображения
        img_features = self.cnn(x['image']).view(x['image'].size(0), -1)

        # Обработка табличных данных
        tab_features = self.tabular_net(x['tabular'])

        # Объединение признаков
        combined = torch.cat([img_features, tab_features], dim=1)

        # Классификация
        return self.classifier(combined)


def calculate_metrics(y_true, y_pred):
    """Вычисление метрик качества"""
    return {
        'auc': roc_auc_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred > 0.5),
        'precision': precision_score(y_true, y_pred > 0.5),
        'recall': recall_score(y_true, y_pred > 0.5)
    }


def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_auc = 0
    history = {'train_loss': [], 'val_metrics': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Перенос данных на устройство
            images = batch['image'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model({'image': images, 'tabular': tabular})
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Валидация
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                tabular = batch['tabular'].to(device)
                labels = batch['label'].cpu().numpy()

                preds = model({'image': images, 'tabular': tabular}).cpu().numpy()
                y_true.extend(labels)
                y_pred.extend(preds)

        # Расчет метрик
        metrics = calculate_metrics(y_true, y_pred)
        scheduler.step(metrics['auc'])

        # Логирование
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {history['train_loss'][-1]:.4f} | "
              f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")

        # Сохранение лучшей модели
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save(model.state_dict(), 'best_model.pt')

    return model, history


if __name__ == "__main__":
    # Пример использования
    image_paths = ["data/sample1.dcm", "data/sample2.dcm"]  # Замените на реальные пути
    tabular_data = pd.DataFrame({
        'age': [45, 60],
        'tumor_size': [12.5, 20.3],
        'biopsy_score': [3, 5]
    })
    labels = [0, 1]

    # Создание датасетов
    dataset = MultimodalCancerDataset(image_paths, tabular_data, labels, augment=True)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42,
                                            stratify=labels)

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Инициализация модели
    model = MultimodalCancerModel(tabular_size=tabular_data.shape[1])

    # Обучение
    trained_model, history = train_model(model, train_loader, val_loader, epochs=10)