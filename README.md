## LSTM Language Model From Scratch

This repository contains a character-level LSTM language model implemented from scratch using PyTorch, trained on the text of Pride and Prejudice.
The project includes:

Data preprocessing\
Vocabulary creation\
Training three model scenarios (Underfitting, Overfitting, Best-Fit)\
Loss curve visualization\
Perplexity evaluation\
Text generation

### 📁 Repository Structure
```text
├── notebooks/
│   └── lstm_language_model.ipynb
│
├── models/
│   └── bestfit_8epoch_model.pth
│
├── plots/
│   ├── underfit_loss.png
│   ├── overfit_loss.png
│   └── bestfit_loss.png
│
├── results/
│   └── losses_summary.csv
│
├── data/
│   └── pride_and_prejudice.txt  
│
└── README.md
```

### 🧠 Model Architecture

A character-level LSTM:\
Embedding dimension: 128\
Hidden dimension: 256\
LSTM Layers: 2\
Dropout: 0.3\
Sequence length: 30 characters\
Optimizer: Adam\
Loss: CrossEntropy

### ✨ Trained Scenarios
Model Type &	Description\
Underfit - Very small model + few epochs\
Overfit -	Large model + many epochs, little regularization\
Best-Fit - Balanced architecture + regularization

The best-fit model is saved in:\
models/bestfit_8epoch_model.pth

### 📈 Training & Evaluation

The notebook performs:

Train/validation split\
Training loops\
Loss tracking\
Perplexity computation\
Plot generation\
Loss plots are available in: plots/

### 🚀 How to Run Training

Open the notebook:\
notebooks/lstm_language_model.ipynb

Run all cells sequentially in:\
Kaggle\
Google Colab\
Local Jupyter Notebook

Dependencies:
```text
pip install torch matplotlib numpy
```
### 🔥 How to Run Inference (Generate Text)

Below is a minimal example of how to load the trained model and generate text:
```text
import torch

# Load checkpoint
checkpoint = torch.load("models/bestfit_8epoch_model.pth", map_location="cpu")
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

# Rebuild model
vocab_size = len(stoi)

class LSTMLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                  dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

model = LSTMLanguageModel(vocab_size, 128, 256, 2, 0.3)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate_text(model, stoi, itos, start="elizabeth ", max_len=300, temperature=0.8):
    fallback = stoi.get(" ", 0)
    tokens = [stoi.get(c.lower(), fallback) for c in start]
    inp = torch.tensor(tokens).unsqueeze(0)
    out = start
    hidden = None

    for _ in range(max_len):
        logits, hidden = model(inp, hidden)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        out += itos[next_token]
        inp = torch.tensor([[next_token]])
    return out

print(generate_text(model, stoi, itos))
```
### 📝 Notes

This is a character-level model → generated text is stylistic but imperfect.\
Word-level or subword tokenization would produce more readable results.\
The notebook contains full training and preprocessing logic.
