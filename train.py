import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import TimeSeriesTransformer, PositionalEncoding
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

def train_model(model, train_loader, val_loader, epochs, lr=0.001):
    criterion = nn.MSELoss()  # Mean Squared Error pour la régression
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Mode entraînement
        running_train_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            print("src shape = ",src.shape)
            print("tgt shape = ",tgt.shape)
            output = model(src, tgt[:, :-1, :])  # Entrée décalée pour le modèle
            print("Output shape:", output.shape)
            loss = criterion(output, tgt[:, 1:, :])
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # Moyenne des pertes sur l'ensemble d'entraînement
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Mode évaluation pour validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt[:, :-1, :])
                val_loss = criterion(output, tgt[:, 1:, :])
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Imprimer les pertes d'entraînement et de validation pour cette époque
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Tracer les courbes de perte d'entraînement et de validation
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length, :]  # Les features d'entrée
        y = self.data[idx+1:idx+self.sequence_length+1, :]  # Valeurs cibles décalées
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Exemple d'utilisation
"""
def create_dataloader(data, sequence_length, batch_size):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length, :-1])  # Les features
        targets.append(data[i:i+sequence_length, 1:])  # Les targets (prix)
    sequences = np.array(sequences)
    targets = np.array(targets)

    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
"""
def create_dataloader(data, sequence_length, batch_size):
    dataset = TimeSeriesDataset(data, sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Séparer les données en ensemble d'entraînement et de validation
def split_data(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size, shuffle=False)  # Garder l'ordre temporel
    return train_data, val_data

# Paramètres des données : suppose que vous avez une série de caractéristiques (features)
# input_size = nombre de features d'entrée, ici par exemple 5 (open, close, high, low, volume)
input_size = 5
sequence_length = 30  # Taille de la fenêtre temporelle utilisée pour les séquences d'entrée
d_model = 64  # Dimension de l'embedding pour chaque timestep
n_heads = 8  # Nombre de têtes d'attention dans chaque bloc
num_encoder_layers = 3  # Nombre de couches de l'encodeur
num_decoder_layers = 3  # Nombre de couches du décodeur
dim_feedforward = 128  # Taille du MLP dans chaque bloc de transformer
dropout = 0.1  # Pour régularisation
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ",device)

# Exemple d'utilisation avec des données fictives
data = np.random.rand(2000, input_size)  # Remplacez par vos données financières

# Créer les DataLoaders

dataset = TimeSeriesDataset(data, sequence_length)
train_data, val_data = split_data(data, test_size=0.2)
train_loader = create_dataloader(train_data, sequence_length, batch_size)
val_loader = create_dataloader(val_data, sequence_length, batch_size)
print(f"Taille du jeu d'entraînement: {len(train_loader.dataset)}")
print(f"Taille du jeu de validation: {len(val_loader.dataset)}")

# Instanciation et entraînement du modèle
model = TimeSeriesTransformer(input_size=input_size, d_model=d_model, n_heads=n_heads,
                              num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                              dim_feedforward=dim_feedforward).to(device)
train_model(model, train_loader, val_loader , epochs=10)


"""
# Exemple avec yfinance
import yfinance as yf

def load_yahoo_finance_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].values  # Choisir les colonnes pertinentes

data = load_yahoo_finance_data('AAPL', '2015-01-01', '2020-01-01')
print(data.shape)

# Charger les données de Yahoo Finance
data = load_yahoo_finance_data('AAPL', '2015-01-01', '2020-01-01')
train_data, val_data = split_data(data, test_size=0.2)
# Créer les DataLoaders
sequence_length = 30
batch_size = 64
train_loader = create_dataloader(train_data, sequence_length, batch_size)
val_loader = create_dataloader(val_data, sequence_length, batch_size)

# Entraînement du modèle
train_model(model, train_loader, epochs=10)"""