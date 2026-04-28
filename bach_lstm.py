import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='bach_measure_dataset.json')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loss_dir', type=str, default='.')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Classical model training on: {device} ---", flush=True)

# 2. Data Loading & Preprocessing
def load_and_split_data(json_path, val_split=0.1):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found at {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lexicon = data['lexicon']
    sentences = data['sentences']
    token_to_idx = {token: i for i, token in enumerate(lexicon)}
    vocab_size = len(lexicon)
    
    X, y = [], []
    for s in sentences:
        indices = [token_to_idx[t] for t in s if t in token_to_idx]
        for i in range(len(indices) - 1):
            X.append(torch.tensor(indices[:i+1]))
            y.append(indices[i+1])
            
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_tensor = torch.tensor(y)
    
    full_dataset = TensorDataset(X_padded, y_tensor)
    
    # Validation Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    return train_ds, val_ds, vocab_size

# 3. Metrics Calculation
def get_metrics(outputs, targets):
    # Accuracy
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == targets).float().mean().item()
    
    # Entropy (Prediction Diversity)
    probs = torch.softmax(outputs, dim=1)
    # Epsilon to avoid log(0)
    ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean().item()
    
    return acc, ent

# 4. Model Architecture
class BachLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(BachLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# Initialization
train_ds, val_ds, vocab_size = load_and_split_data(args.data_path)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

model = BachLSTM(vocab_size, embed_dim=32, hidden_dim=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# History for Plotting
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'perplexity': []
}

# 5. Training Loop
def train_classical(epochs=args.epochs):
    print(f"Starting Training: {epochs} epochs", flush=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, train_ent = 0, 0, 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            acc, ent = get_metrics(output, target)
            train_loss += loss.item()
            train_acc += acc
            train_ent += ent
            
        # Validation Phase
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                acc, _ = get_metrics(output, target)
                val_loss += loss.item()
                val_acc += acc
        
        # Calculate Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        perplexity = np.exp(avg_train_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc / len(train_loader))
        history['val_acc'].append(avg_val_acc)
        history['perplexity'].append(perplexity)
        
        # Cluster Log: Her epoch sonunda tek satır
        print(f"Epoch [{epoch+1:3d}/{epochs}] "
              f"Loss: {avg_train_loss:.3f}/{avg_val_loss:.3f} | "
              f"Acc: {avg_val_acc:.2%} | "
              f"PPL: {perplexity:.2f} | "
              f"Ent: {train_ent/len(train_loader):.3f}", flush=True)

    # Save Everything
    save_outputs()

def save_outputs():
    # 1. Model
    torch.save(model.state_dict(), "bach_lstm_model.pt")
    
    # 2. Plot: Loss & Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.loss_dir, 'training_metrics.png'))
    print(f"--- Training Complete. Files saved in {args.loss_dir} ---", flush=True)

if __name__ == "__main__":
    train_classical()