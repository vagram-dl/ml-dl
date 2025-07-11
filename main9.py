import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


raw_data = [
 ("Искусственный интеллект меняет мир.", "AI меняет мир."),
    ("Машинное обучение — мощный инструмент анализа данных.", "ML помогает анализировать данные."),
    ("Современные технологии развиваются очень быстро.", "Технологии быстро развиваются."),
    ("Будущее за искусственным интеллектом и роботами.", "ИИ и роботы — будущее."),

]

src_tokenizer = get_tokenizer("basic_english")
tgt_tokenizer = get_tokenizer("basic_english")

def yield_tokens(data, tokenizer):
    for src, tgt in data:
        yield tokenizer(src)
        yield tokenizer(tgt)

vocab = build_vocab_from_iterator(
    yield_tokens(raw_data, src_tokenizer),
    specials=["<unk>", "<pad>", "<sos>", "<eos>"]
)
vocab.set_default_index(vocab["<unk>"])

def numericalize(text, tokenizer, vocab):
    tokens = tokenizer(text)
    return torch.tensor(
        [vocab["<sos>"]] + [vocab[token] for token in tokens] + [vocab["<eos>"]],
        dtype=torch.long
    )


class Seq2SeqSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, bidirectional=False)
        self.decoder = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        encoder_out, (hidden, cell) = self.encoder(src_emb)

        tgt_emb = self.embedding(tgt)
        decoder_out, _ = self.decoder(tgt_emb, (hidden, cell))

        logits = self.fc(decoder_out)
        return logits


vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
learning_rate = 0.001
num_epochs = 20
batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqSummarizer(vocab_size, embed_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src_text, tgt_text in raw_data:
        src = numericalize(src_text, src_tokenizer, vocab).unsqueeze(1).to(device)
        tgt = numericalize(tgt_text, tgt_tokenizer, vocab).unsqueeze(1).to(device)

        outputs = model(src, tgt[:-1])
        loss = criterion(outputs.view(-1, vocab_size), tgt[1:].view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")


def generate_summary(model, input_text, max_len=10):
    model.eval()
    src = numericalize(input_text, src_tokenizer, vocab).unsqueeze(1).to(device)

    with torch.no_grad():
        src_emb = model.embedding(src)
        encoder_out, (hidden, cell) = model.encoder(src_emb)

        decoder_input = torch.tensor([vocab["<sos>"]], device=device).unsqueeze(0)
        decoded_words = []

        for _ in range(max_len):
            tgt_emb = model.embedding(decoder_input)
            output, (hidden, cell) = model.decoder(tgt_emb, (hidden, cell))
            logits = model.fc(output)
            pred = logits.argmax(dim=-1)

            word = vocab.get_itos()[pred.item()]
            if word == "<eos>":
                break
            decoded_words.append(word)
            decoder_input = pred

    return " ".join(decoded_words)


input_text = "Искусственный интеллект меняет мир."
summary = generate_summary(model, input_text)
print("\nOriginal:", input_text)
print("Summary:", summary)