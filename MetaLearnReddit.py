import torch.nn as nn


#JUST FOR VIEWING PURPOSES. API FETCH PART IS DELETED DUE TO CONFIDENTIAL INFORMATION (API KEY, API ID)
#SECRET KEY AND ID ARE PROTECTED

class WordBiasLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super(WordBiasLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        probs = self.sigmoid(logits)
        return probs.squeeze(-1)  # shape: [batch_size, sequence_length]


def train_model(model, dataloader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")


def predict_biased_words(model, text, word_to_ix):
    model.eval()
    words, _ = label_words_from_article(text, right_biased, left_biased, neutral_or_both)
    idxs = [word_to_ix.get(w, 1) for w in words]
    x = torch.tensor([idxs[:30] + [0]*(30-len(idxs))])
    with torch.no_grad():
        preds = model(x).squeeze().numpy()
    return [w for w, p in zip(words, preds) if p > 0.5]



# Replace with your actual article text
text = input()
words, labels = label_words_from_article(text, right_biased, left_biased, neutral_or_both)
vocab = build_vocab(words)
dataset = BiasDataset(words, labels, vocab)
loader = DataLoader(dataset, batch_size=1)

model = WordBiasLSTM(vocab_size=len(vocab))
train_model(model, loader)

# Predict
print("This is an unsupervised version of the model, made in PyTorch.")
biased_words = predict_biased_words(model, text, vocab)
print("Biased words detected:", biased_words)

