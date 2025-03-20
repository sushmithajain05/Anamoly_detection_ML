import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Load Dataset
file_path = "EB_2021.csv"
df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.dropna(subset=["power_load"])

# Normalize Data
scaler = MinMaxScaler()
df["normalized_power"] = scaler.fit_transform(df[["power_load"]])

# Create Time Series Sequences
sequence_length = 100

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i: i + seq_length])
    return np.array(sequences)

sequences = create_sequences(df["normalized_power"].values, sequence_length)

# Define PyTorch Dataset
class PowerLoadDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

train_dataset = PowerLoadDataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define SR-CCN Model
class SRCCN(nn.Module):
    def __init__(self):
        super(SRCCN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Training Function
def train_srccn(model, dataloader, epochs=10, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.unsqueeze(1)  # Add channel dimension
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Train Model
model = SRCCN()
train_srccn(model, train_loader, epochs=20)

# Save Model
torch.save(model.state_dict(), "srccn_model.pth")
