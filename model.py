import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        # Embedding layer pour convertir les séries temporelles en vecteurs de taille d_model 
        self.embedding = nn.Linear(input_size, d_model)
        
        # Encodage des positions pour garder la notion de temps
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Encoder et decoder des transformers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layer pour faire la prédiction
        self.fc_out = nn.Linear(d_model, input_size)  # Prédiction de la valeur future (prix)
        print("init ok")

    def forward(self, src, tgt):
        # src = entrée (historique des prix, etc.), taille : (batch_size, sequence_length, input_size)
        # tgt = cible (prix à prédire), taille : (batch_size, target_sequence_length, input_size)
        
        # Embedding des entrées
        src = self.embedding(src)  # Transformer l'entrée en vecteur de taille d_model
        print(src.shape)
        src = self.positional_encoding(src)  # Ajouter l'encodage positionnel
        print(src.shape)
        # Embedding des cibles
        tgt = self.embedding(tgt)
        print(src.shape)
        tgt = self.positional_encoding(tgt)
        print(src.shape)
        # Transformer attend une entrée de forme (sequence_length, batch_size, d_model)
        src = src.permute(1, 0, 2)  # (sequence_length, batch_size, d_model)
        tgt = tgt.permute(1, 0, 2)  # (sequence_length, batch_size, d_model)
        print(src.shape)
        # Passer par le modèle Transformer
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        print(src.shape)
        # Remettre l'ordre à (batch_size, sequence_length, d_model)
        output = output.permute(1, 0, 2)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Calcul de l'encodage positionnel (basé sur une combinaison de sin et cos)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

