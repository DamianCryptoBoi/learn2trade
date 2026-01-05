import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

class CryptoDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TradingNet(nn.Module):
    def __init__(self, input_dim):
        super(TradingNet, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)
        
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout3(x)
        return self.output(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerTradingNet(nn.Module):
    def __init__(self, input_dim, d_model=1024, nhead=16, num_layers=12, dropout=0.1):
        super(TransformerTradingNet, self).__init__()
        
        # Project input features to d_model dimension
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        # dim_feedforward is typically 4 * d_model
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4 * d_model, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output Head
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(), # GELU is standard for modern Transformers
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        
        # 1. Embed and add position info
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        # 2. Pass through Transformer
        x = self.transformer_encoder(x)
        
        # 3. Global Average Pooling
        x = x.mean(dim=1)
        
        # 4. Classification
        x = self.decoder(x)
        return x

