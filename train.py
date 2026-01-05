import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib
import os
from src.model import CryptoDataset, TradingNet

def train_model():
    def compute_classification_metrics(probs, targets, threshold=0.5):
        preds = (probs >= threshold).astype(float)
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
            "f1": f1_score(targets, preds, zero_division=0),
        }
        # Guard AUCs when only one class is present.
        if len(np.unique(targets)) > 1:
            metrics["roc_auc"] = roc_auc_score(targets, probs)
            metrics["pr_auc"] = average_precision_score(targets, probs)
        else:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
        return metrics

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for batch_features, batch_targets in loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets.unsqueeze(1))
                total_loss += loss.item()
                probs = torch.sigmoid(outputs).squeeze(1)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        avg_loss = total_loss / len(loader)
        return avg_loss, np.array(all_probs), np.array(all_targets)

    # Load Data
    print("Loading data...")
    train_df = pd.read_csv('data/btc_features_train.csv')
    test_df = pd.read_csv('data/btc_features_test.csv')
    
    # Feature Selection
    # Drop non-feature columns
    drop_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'target']
    
    # Filter only numeric columns that are not in drop_cols
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].values
    y = train_df['target'].values

    # Validation split (stratified to preserve class balance)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    
    # Datasets & DataLoaders
    train_dataset = CryptoDataset(X_train_scaled, y_train)
    val_dataset = CryptoDataset(X_val_scaled, y_val)
    test_dataset = CryptoDataset(X_test_scaled, y_test)
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    input_dim = len(feature_cols)
    model = TradingNet(input_dim)
    
    # CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # On Mac M1/M2/M3
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    print(f"Using device: {device}")
    model.to(device)
    
    # Class weighting from current train split
    pos_count = max(float(y_train.sum()), 1.0)
    neg_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 200
    patience = 20
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_threshold = 0.5
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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
        val_loss, val_probs, val_targets = evaluate(val_loader)
        val_metrics = compute_classification_metrics(val_probs, val_targets, threshold=0.5)

        # Simple threshold sweep on validation to maximize F1
        thresholds = np.linspace(0.1, 0.9, 9)
        best_f1 = val_metrics["f1"]
        best_threshold_epoch = 0.5
        for thr in thresholds:
            candidate_metrics = compute_classification_metrics(val_probs, val_targets, threshold=thr)
            if candidate_metrics["f1"] > best_f1:
                best_f1 = candidate_metrics["f1"]
                best_threshold_epoch = thr
        if best_f1 > val_metrics["f1"]:
            val_metrics["f1"] = best_f1
            best_threshold = best_threshold_epoch

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | RocAUC: {val_metrics['roc_auc']:.4f} | "
                f"PR-AUC: {val_metrics['pr_auc']:.4f} | Thr*: {best_threshold:.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), 'model/best_model.pth')
            print(f"Epoch {epoch+1}: New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch}).")
                break

    # Final test evaluation using best checkpoint and best threshold from validation
    model.load_state_dict(torch.load('model/best_model.pth', map_location=device))
    model.to(device)
    test_loss, test_probs, test_targets = evaluate(test_loader)
    test_metrics = compute_classification_metrics(test_probs, test_targets, threshold=best_threshold)
    print(
        f"Test | Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']:.4f} | "
        f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | RocAUC: {test_metrics['roc_auc']:.4f} | "
        f"PR-AUC: {test_metrics['pr_auc']:.4f} | Thr*: {best_threshold:.2f}"
    )

if __name__ == "__main__":
    train_model()
