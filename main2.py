import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt5.QtGui import QFont
from transformers import T5Tokenizer, T5ForConditionalGeneration

#  Класс агента с ruT5
class ImprovedAgent:
    def __init__(self):
        self.memory = []
        self.model_name = "ai-forever/ruT5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def remember(self, user_input, bot_response):
        self.memory.append({"user": user_input, "bot": bot_response})
        if len(self.memory) > 5:
            self.memory.pop(0)

    def think(self, user_input):
        prompt = self.build_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.remember(user_input, response)
        return response

    def build_prompt(self,user_input):
        prompt = """Ты — опытный ассистент. Отвечай ясно, кратко и точно."""
        if self.memory:
            prompt += "\n\nИстория:\n"
            for entry in self.memory[-3:]:
                prompt += f"Пользователь: {entry['user']}\n"
                prompt += f"Агент: {entry['bot']}\n"

        prompt += f"\nПользователь: {user_input}\n"
        prompt += "Агент: "
        return prompt

        if self.memory:
            prompt += "История переписки:\n"
            for entry in self.memory[-3:]:
                prompt += f"Пользователь: {entry['user']}\n"
                prompt += f"Агент: {entry['bot']}\n"

        prompt += f"Пользователь: {user_input}\n"
        prompt += "Агент:"
        return prompt.strip()



class ChatWindow(QWidget):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.setWindowTitle("🤖 ИИ-агент на основе ruT5")
        self.resize(900, 600)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        # История чата
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setFont(QFont("Arial", 14))
        self.chat_log.setStyleSheet("""
            font-size: 14pt;
            padding: 10px;
            border-radius: 8px;
            background-color: #f0f0f0;
        """)
        layout.addWidget(self.chat_log)

        # Поле ввода
        input_layout = QHBoxLayout()

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Введите ваш вопрос...")
        self.input_box.setFont(QFont("Arial", 14))
        self.input_box.setStyleSheet("""
            height: 50px;
            font-size: 14pt;
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 8px;
        """)
        self.input_box.returnPressed.connect(self.on_send)
        input_layout.addWidget(self.input_box)

        # Кнопка отправки
        self.send_button = QPushButton("📩 Отправить")
        self.send_button.setFont(QFont("Arial", 14))
        self.send_button.setFixedWidth(150)
        self.send_button.setStyleSheet("""
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            font-size: 14pt;
            padding: 10px;
            border-radius: 8px;
        """)
        self.send_button.clicked.connect(self.on_send)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)
        self.setLayout(layout)

    def on_send(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return

        self.chat_log.append(f"<b>Вы:</b> {user_input}")
        self.input_box.clear()

        response = self.agent.think(user_input)
        self.chat_log.append(f"<b>Бот:</b> {response}")

        # Прокрутка вниз
        self.chat_log.verticalScrollBar().setValue(
            self.chat_log.verticalScrollBar().maximum()
        )



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 12))

    print(" Загружаем ruT5...")
    agent = ImprovedAgent()
    window = ChatWindow(agent)
    window.show()

    sys.exit(app.exec_())