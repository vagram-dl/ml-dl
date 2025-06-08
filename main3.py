import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import telebot

# 🔹 Загрузка модели
model_name = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Если нет pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

memory = []

def remember(user_input, bot_response):
    memory.append({"user": user_input, "bot": bot_response})
    if len(memory) > 5:
        memory.pop(0)

def generate_response(prompt):
    full_prompt = f"Пользователь: {prompt}\nАгент: "

    if memory:
        history = "\n".join([f"Пользователь: {entry['user']}\nАгент: {entry['bot']}" for entry in memory[-3:]])
        full_prompt = f"{history}\n\n{full_prompt}"

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # ← теперь здесь нет ошибки
        max_new_tokens=150,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    response = response[len(full_prompt):].strip()

    remember(prompt, response)
    return response


# 🤖 Telegram-бот
BOT_TOKEN = '8192023657:AAHdyH4LG0q6t7atNtG13HKfg_NJV7fZVG0'
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    response = generate_response(user_input)
    bot.reply_to(message, response)

if __name__ == "__main__":
    print(" Бот запущен!")
    bot.polling(none_stop=True)