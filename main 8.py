# рекомендательная система

import  sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QLabel,QLineEdit,QPushButton,QListWidget
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data():
    ratings_path = 'data/ml-100k/u.data'
    movies_path = 'data/ml-100k/u.item'

    ratings_df = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    ratings_df = ratings_df.drop(columns=['timestamp'])

    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                     'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None)
    movies_df.columns = ['item_id', 'title'] + genre_columns
    movies_df = movies_df[['item_id', 'title'] + genre_columns]

    ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes
    ratings_df['item_id'] = ratings_df['item_id'].astype('category').cat.codes

    return ratings_df, movies_df[genre_columns], movies_df[['item_id', 'title']]


class RatingDataset(Dataset):
    def __init__(self, df, genres):
        self.df = df
        self.genres = genres

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = row['item_id']
        genre_vector = torch.tensor(self.genres.iloc[item_id].values, dtype=torch.float32)
        return (
            torch.tensor(row['user_id'], dtype=torch.long),
            torch.tensor(row['item_id'], dtype=torch.long),
            torch.tensor(row['rating'], dtype=torch.float32),
            genre_vector
        )

class NeuMFWithContent(nn.Module):
    def __init__(self,num_users,num_items,embedding_dim=64,content_dim=18):
        super().__init__()
        self.user_embedding=nn.Embedding(num_users,embedding_dim)
        self.item_embedding=nn.Embedding(num_items,embedding_dim)
        self.content_proj=nn.Linear(content_dim,embedding_dim)

        self.fc_layers=nn.Sequential(
            nn.Linear(embedding_dim * 3,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)


    def forward(self, user_ids, item_ids, content):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        content_emb = self.content_proj(content)
        x = torch.cat([user_emb, item_emb, content_emb], dim=1)
        return self.fc_layers(x).flatten()


def train_model(ratings_df, genres_df):
    print("Обучение модели...")

    train_df, _ = train_test_split(ratings_df, test_size=0.2, random_state=42)
    train_dataset = RatingDataset(train_df, genres_df)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['item_id'].nunique()
    embedding_dim = 64
    content_dim = genres_df.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMFWithContent(num_users, num_items, embedding_dim, content_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for user_ids, item_ids, ratings, genres in train_loader:
            user_ids, item_ids, ratings, genres = (
                user_ids.to(device),
                item_ids.to(device),
                ratings.to(device),
                genres.to(device)
            )

            optimizer.zero_grad()
            outputs = model(user_ids, item_ids, genres)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10,Loss:{total_loss:.4f}")

    torch.save(model.state_dict(),"neumf_with_content.pth")
    print("Модель сохранена")
    return model

def predict_raiting(model,user_id,item_id,genres_df,device):
    model.eval()
    with torch.no_grad():
        genre_vector=torch.tensor(genres_df.iloc[item_id].values,dtype=torch.float32).to(device)
        user_tensor=torch.tensor([user_id],dtype=torch.long).to(device)
        item_tensor=torch.tensor([item_id],dtype=torch.long).to(device)
        rating=model(user_tensor,item_tensor,genre_vector.unsqueeze(0))
    return rating.item()

def recommend_top_n(model,user_id,movies_df,genres_df,n=10):
    device=next(model.parameters()).device
    ratings=[]
    for idx in range(len(movies_df)):
        item_id=movies_df.iloc[idx]['item_id']
        rating=predict_rating(model, user_id, item_id, genres_df, device)
        ratings.append((item_id,rating))
    ratings.sort(key=lambda x:x[1],reverse=True)
    top_n=ratings[:n]
    recommended_movies = [movies_df[movies_df['item_id'] == item_id]['title'].values[0] for item_id, _ in top_n]
    return recommend_movies

class RecommenderApp(QWidget):
    def __init__(self,model,movies_df,genres_df):
        super().__init__()
        self.model=model
        self.movies_df=movies_df
        self.genres_df=genres_df
        self.device=next(model.parameters()).device
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Рекомендательная система')
        layout=QVBoxLayout()

        self.user_input=QLineEdit(self)
        self.user_input.setPlaceholderText("Введите ID пользователя")

        btn_recommend=QPushButton("Получить рекомендации",self)
        btn_recommend.clicked.connect(self.show_recommendations)

        self.list_widget = QListWidget(self)

        layout.addWidget(QLabel("Введите ID пользователя:"))
        layout.addWidget(self.user_input)
        layout.addWidget(btn_recommend)
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def show_recommendations(self):
        try:
            user_id = int(self.user_input.text())
            recommendations = recommend_top_n(self.model, user_id, self.movies_df, self.genres_df, n=10)
            self.list_widget.clear()
            self.list_widget.addItems(recommendations)
        except Exception as e:
            self.list_widget.clear()
            self.list_widget.addItem(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    ratings_df, genres_df, movies_df = load_data()

    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['item_id'].nunique()
    embedding_dim = 64
    content_dim = genres_df.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuMFWithContent(num_users, num_items, embedding_dim, content_dim).to(device)
    if os.path.exists("neumf_with_content.pth"):
        model.load_state_dict(torch.load("neumf_with_content.pth", map_location=device))
        print("Модель загружена.")
    else:
        model = train_model(ratings_df, genres_df)

    app = QApplication(sys.argv)
    window = RecommenderApp(model, movies_df, genres_df)
    window.show()
    sys.exit(app.exec_())