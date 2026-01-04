import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os
from src.model import CryptoDataset, TransformerTradingNet

def create_sequences(features, targets, seq_len):
    """
    Creates sequences of length seq_len from features.
    Target is the target at the end of the sequence.
    """
    xs, ys = [], []
    # Use stride tricks or simple loop. Loop is safer for now.
    # We need to predict target at i, using features from i-seq_len to i
    # So if seq_len=60, first prediction is at index 59 (using 0-59)
    
    # Optimization: Use numpy stride tricks for speed
    num_samples = len(features) - seq_len
    
    # Create indices
    indices = np.arange(num_samples)[:, None] + np.arange(seq_len)[None, :]
    
    xs = features[indices]
    ys = targets[seq_len:]
    
    return xs, ys

def train_model():
    # Load Data
    print("Loading data...")
    train_df = pd.read_csv('data/btc_features_train.csv')
    test_df = pd.read_csv('data/btc_features_test.csv')
    
    # Feature Selection
    # Drop non-feature columns
    drop_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'target']
    
    # Filter only numeric columns that are not in drop_cols
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    
    # Create Sequences
    SEQ_LEN = 60 # 15 hours of context
    print(f"Creating sequences (SEQ_LEN={SEQ_LEN})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQ_LEN)
    
    print(f"Train shape: {X_train_seq.shape}")
    
    # Datasets & DataLoaders
    train_dataset = CryptoDataset(X_train_seq, y_train_seq)
    test_dataset = CryptoDataset(X_test_seq, y_test_seq)
    
    # Optimization: Increased batch size and added DataLoader workers
    batch_size = 4096 # Reduced slightly from 16k because sequences are larger
    
    # num_workers=4: Parallelize data loading
    # pin_memory=True: Faster host-to-device transfer
    # persistent_workers=True: Keep workers alive between epochs
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # Model Setup
    input_dim = len(feature_cols)
    # Switch to Transformer
    model = TransformerTradingNet(input_dim, d_model=128, nhead=4, num_layers=4)
    
    # CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # On Mac M1/M2/M3
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    print(f"Using device: {device}")
    model.to(device)
    
    # Class weighting
    # Positive rate ~ 0.37 (from features.py output) -> Neg/Pos ~ 63/37 ~ 1.7
    pos_weight = torch.tensor([1.7]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10000
    best_loss = float('inf')
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets.unsqueeze(1))
                test_loss += loss.item()
                
                # Apply sigmoid for metrics since model returns logits now
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {accuracy:.4f} | Precision: {precision:.4f}")
        
        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            # Create model directory if not exists
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), 'model/best_model.pth')
            print(f"Epoch {epoch+1}: New best model saved! (Loss: {avg_test_loss:.4f})")

if __name__ == "__main__":
    train_model()
