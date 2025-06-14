import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from PyQt5.QtWidgets import (QApplication,QMainWindow,QVBoxLayout,QHBoxLayout,
                             QTextEdit,QPushButton,QWidget,QLabel)
from PyQt5.QtCore import Qt
import sys






def tokenize(text):
    """Разбиваем текст на токены (слова)"""
    return text.lower().split()


def build_vocab(sentences, min_freq=1):
    """Строим словарь из предложений"""
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence))

    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    idx = len(vocab)

    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1

    return vocab


def encode(tokens, vocab, add_bos=False, add_eos=False):
    """Кодируем токены в индексы с возможностью добавления служебных токенов"""
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if add_bos:
        ids = [vocab["<bos>"]] + ids
    if add_eos:
        ids = ids + [vocab["<eos>"]]
    return ids


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        # Энкодер
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Декодер
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, y=None):
        # Энкодер
        x_emb = self.encoder_embedding(x)
        _, (h, c) = self.lstm_encoder(x_emb)

        # Декодер
        if y is not None:
            y_emb = self.decoder_embedding(y)
            dec_out, _ = self.lstm_decoder(y_emb, (h, c))
            logits = self.output(dec_out)
            return logits
        else:
            # Генерация (не используется в обучении)
            bos = torch.tensor([[2]]).to(x.device)
            generated_tokens = []

            dec_input = bos
            for _ in range(30):
                dec_out, (h, c) = self.lstm_decoder(
                    self.decoder_embedding(dec_input), (h, c))
                logit = self.output(dec_out[:, -1, :])
                next_token = logit.argmax(-1)
                generated_tokens.append(next_token.item())

                if next_token.item() == 3:  # <eos>
                    break
                dec_input = next_token.unsqueeze(0)

            return torch.tensor(generated_tokens)


def train_model(model, vocab, sentence_pairs, num_epochs=50):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    print("Начинаем обучение...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_text, target_text in sentence_pairs:
            # Кодируем вход энкодера
            input_ids = torch.tensor(encode(tokenize(input_text), vocab)).unsqueeze(0)

            # Подготовка входа и выхода декодера
            target_tokens = tokenize(target_text)
            decoder_input = torch.tensor(encode(target_tokens, vocab, add_bos=True)).unsqueeze(0)
            target_output = torch.tensor(encode(target_tokens, vocab, add_eos=True)).unsqueeze(0)

            # Forward pass
            logits = model(input_ids, decoder_input)

            # Вычисляем loss
            loss = criterion(logits.view(-1, len(vocab)), target_output.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(sentence_pairs)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Промежуточная генерация для мониторинга
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_inputs = ["привет", "как жизнь?", "что нового?"]
            for inp in test_inputs:
                response = generate_with_beam_search(model, inp, vocab)
                print(f"Тест: '{inp}' -> '{response}'")
            model.train()


def generate_with_beam_search(model, input_text, vocab, beam_width=5, max_len=30):
    """Генерация ответа с использованием beam search"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        tokens = tokenize(input_text)
        ids = encode(tokens, vocab)
        src_tensor = torch.tensor(ids).unsqueeze(0).to(device)

        x_emb = model.encoder_embedding(src_tensor)
        _, (h, c) = model.lstm_encoder(x_emb)

        beams = [([vocab["<bos>"]], 0.0, h, c)]

        for _ in range(max_len):
            new_beams = []

            for seq, score, h_beam, c_beam in beams:
                if seq[-1] == vocab["<eos>"]:
                    new_beams.append((seq, score, h_beam, c_beam))
                    continue

                last_token = torch.tensor([[seq[-1]]]).to(device)
                h_beam = h_beam.clone()
                c_beam = c_beam.clone()

                dec_emb = model.decoder_embedding(last_token)
                dec_out, (h_new, c_new) = model.lstm_decoder(dec_emb, (h_beam, c_beam))
                logit = model.output(dec_out[:, -1, :])

                log_probs = torch.nn.functional.log_softmax(logit, dim=-1)
                top_scores, top_indices = log_probs.topk(beam_width)

                for i in range(beam_width):
                    next_token = top_indices[0][i].item()
                    new_seq = seq + [next_token]
                    new_score = score + top_scores[0][i].item()
                    new_beams.append((new_seq, new_score, h_new, c_new))

            new_beams.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)  # Смягченная нормировка
            beams = new_beams[:beam_width]

            if all(beam[0][-1] == vocab["<eos>"] for beam in beams):
                break

        best_seq, best_score, _, _ = max(beams, key=lambda x: x[1] / len(x[0]))

        index_to_word = {idx: word for word, idx in vocab.items()}
        return " ".join([index_to_word[i] for i in best_seq if i not in (vocab["<bos>"], vocab["<eos>"])])


if __name__ == "__main__":
    sentence_pairs = [
        ("привет", "привет! как дела?"),
        ("здравствуйте", "рад вас видеть"),
        ("как жизнь?", "всё хорошо, спасибо"),
        ("что нового?", "ничего особенного"),
        ("этот фильм потрясающий", "мне очень понравилось кино"),
        ("плохое качество звука", "фильм был ужасным"),
        ("расскажи анекдот", "у меня плохая память на шутки"),
        ("ты умеешь говорить", "конечно, я же AI"),
        ("где купить билеты", "через официальный сайт"),
        ("покажи мне кино", "посмотри фильмы недели"),
        ("как жизнь?", "всё нормально"),
        ("как жизнь?", "живу, работаю, учусь"),
        ("что нового?", "немного отдыхаю"),
        ("расскажи анекдот", "зачем роботу юмор?"),
        ("отличная погода", "да, сегодня прекрасный день"),
        ("что планируешь", "буду изучать нейронные сети"),
        ("любимое блюдо", "обожаю пасту с сыром"),
        ("как тебя зовут", "я языковая модель"),
        ("расскажи шутку", "программист звонит в поддержку: у вас баг в программе..."),
        ("совет на день", "никогда не переставай учиться"),
        ("лучший фильм", "для меня это 'Начало'"),
        ("пока", "до свидания! хорошего дня")
    ]

    all_sentences = [s[0] for s in sentence_pairs] + [s[1] for s in sentence_pairs]
    vocab = build_vocab(all_sentences)
    print(f"Словарь построен, размер: {len(vocab)} токенов")

    model = Seq2SeqModel(len(vocab))
    print("Модель создана")

    train_model(model, vocab, sentence_pairs, num_epochs=50)

    test_phrases = [
        "привет",
        "как дела",
        "что нового",
        "расскажи шутку",
        "какой твой любимый фильм"
    ]

    print("\nТестирование модели:")
    for phrase in test_phrases:
        response = generate_with_beam_search(model, phrase, vocab)
        print(f"Вход: '{phrase}' -> Ответ: '{response}'")


def load_model_and_vocab():
    """Загрузка или обучение модели с гарантированным возвратом (model, vocab)"""
    try:
        # 1. Подготовка данных (добавьте больше примеров)
        sentence_pairs = [
            ("привет", "привет! как дела?"),
            ("как жизнь?", "всё хорошо, спасибо"),
            ("что нового?", "ничего особенного"),
            ("расскажи анекдот", "программист звонит в поддержку..."),
            ("какой твой любимый фильм", "для меня это 'Начало'")
        ]

        # 2. Построение словаря
        all_sentences = [s[0] for s in sentence_pairs] + [s[1] for s in sentence_pairs]
        vocab = build_vocab(all_sentences)

        # 3. Создание и обучение модели
        model = Seq2SeqModel(len(vocab))
        train_model(model, vocab, sentence_pairs)

        return model, vocab  # Важно: возвращаем кортеж

    except Exception as e:
        print(f"Ошибка в load_model_and_vocab(): {str(e)}")
        # Возвращаем заглушки в случае ошибки
        dummy_vocab = {"<pad>": 0, "<unk>": 1}
        dummy_model = Seq2SeqModel(len(dummy_vocab))
        return dummy_model, dummy_vocab

class ChatBotGUI(QMainWindow):
    def __init__(self,model,vocab):
        super().__init__()
        self.model=model
        self.vocab=vocab

        self.setWindowTitle("Seq2Seq ChatBot")
        self.setFixedSize(600,500)

        central_widget=QWidget()
        self.setCentralWidget(central_widget)
        main_layout=QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.input_label=QLabel("Ваше сообщение:")
        self.input_field=QTextEdit()
        self.input_field.setMaximumHeight(100)

        self.generate_btn=QPushButton("Сгенерировать ответ")
        self.generate_btn.clicked.connect(self.generate_response)

        self.output_label=QLabel("Ответ бота :")
        self.output_field=QTextEdit()
        self.output_field.setReadOnly(True)

        main_layout.addWidget(self.input_label)
        main_layout.addWidget(self.input_field)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        main_layout.addWidget(self.output_label)
        main_layout.addWidget(self.output_field)

        self.setStyleSheet("""
            QTextEdit{
                border: 1 px solid #ccc; 
                border_radius: 5 px;
                padding:8px;
                font_size:14 px;
                }
                QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #45a049;
            }""")
    def generate_response(self):
        try:
            input_text=self.input_field.toPlainText().strip()

            if not input_text:
                self.output_field.setPlainText("Введение сообщение !")
                return
            response=generate_with_beam_search(self.model,input_text,self.vocab)

            self.output_field.setPlainText(response)
        except Exception as e:
            self.output_field.setPlainText(f"Ошибка: {str(e)}")

if __name__=="__main__":
    model,vocab=load_model_and_vocab()

    app=QApplication(sys.argv)
    window=ChatBotGUI(model,vocab)
    window.show()
    sys.exit(app.exec_())