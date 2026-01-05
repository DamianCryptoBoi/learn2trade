import torch
import pandas as pd
import numpy as np
import joblib
from src.model import TransformerTradingNet
import matplotlib.pyplot as plt

def create_sequences(features, seq_len):
    """
    Creates sequences of length seq_len from features.
    """
    num_samples = len(features) - seq_len
    indices = np.arange(num_samples)[:, None] + np.arange(seq_len)[None, :]
    xs = features[indices]
    return xs

def backtest_model():
    print("Loading test data and model...")
    # Load raw test data (with prices) to simulate trades
    # We need the original timestamps and prices, not just features
    # But we also need the scaled features for the model
    
    # Reload processed test features just for X input
    test_features_df = pd.read_csv('data/btc_features_test.csv')
    
    # We need to reconstruct the raw prices context. 
    # Since we saved features/targets but dropped some columns in train.py local vars, 
    # we should have kept them or re-merge.
    # Actually, `btc_features_test.csv` has 'close', 'high', 'low' etc because in `src/features.py` we didn't drop them, we just added columns.
    # Let's verify `src/features.py`: `df` returned has all original columns + indicators. We saved that to csv.
    
    df = pd.read_csv('data/btc_features_test.csv')
    
    # Prepare features for model
    drop_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X_test = df[feature_cols].values
    
    # Load Scaler
    scaler = joblib.load('model/scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    
    # Create Sequences
    SEQ_LEN = 128 # Updated to match train.py
    X_test_seq = create_sequences(X_test_scaled, SEQ_LEN)
    
    # Adjust df to match sequence length (remove first SEQ_LEN rows)
    df = df.iloc[SEQ_LEN:].reset_index(drop=True)
    
    # Load Model
    input_dim = len(feature_cols)
    # Updated architecture to match train.py (Speed config)
    model = TransformerTradingNet(input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load state dict and handle torch.compile prefix
    state_dict = torch.load('model/best_model.pth', map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v # Remove '_orig_mod.'
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Get Predictions
    # Process in batches to avoid OOM if dataset is huge
    batch_size = 4096
    probs = []
    
    X_tensor = torch.FloatTensor(X_test_seq)
    num_batches = int(np.ceil(len(X_tensor) / batch_size))
    
    print(f"Predicting in {num_batches} batches...")
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_X = X_tensor[i*batch_size : (i+1)*batch_size].to(device)
            logits = model(batch_X)
            batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(batch_probs)
            
    df['prob'] = probs
    
    # Backtest Loop
    # Strategy: 
    # If prob > threshold, Buy.
    # Exit: TP (+1%) or SL (-1%) or Time Limit (32 candles = 8 hours).
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    print("\n--- Backtest Results ---")
    
    for thresh in thresholds:
        balance = 1000.0  # Initial Capital
        trades = []
        
        # Simple simulation: assume we can only be in one trade at a time
        in_trade = False
        entry_price = 0
        entry_idx = 0
        
        # Vectorized simulation is hard with "one trade at a time", loop is safer
        # But loop is slow. Given test set size (1 year ~ 35k rows), it's manageable.
        
        for i in range(len(df)):
            if in_trade:
                # Check exit conditions
                curr_low = df.loc[i, 'low']
                curr_high = df.loc[i, 'high']
                curr_close = df.loc[i, 'close']
                
                tp_price = entry_price * 1.01
                sl_price = entry_price * 0.99
                
                pnl = 0
                exit_reason = ""
                
                if curr_low <= sl_price:
                    pnl = -0.01
                    balance *= (1 + pnl)
                    in_trade = False
                    exit_reason = "SL"
                elif curr_high >= tp_price:
                    pnl = 0.01
                    balance *= (1 + pnl)
                    in_trade = False
                    exit_reason = "TP"
                elif i - entry_idx >= 32: # Time limit 8h
                    pnl = (curr_close - entry_price) / entry_price
                    balance *= (1 + pnl)
                    in_trade = False
                    exit_reason = "Time"
                
                if not in_trade:
                    trades.append({'pnl': pnl, 'reason': exit_reason})
                    
            elif df.loc[i, 'prob'] > thresh:
                # Enter Trade
                in_trade = True
                entry_price = df.loc[i, 'close']
                entry_idx = i
        
        total_return = (balance - 1000) / 1000 * 100
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
        results[thresh] = total_return
        print(f"Threshold: {thresh} | Trades: {len(trades)} | Win Rate: {win_rate:.2%} | Final Balance: ${balance:.2f} | Return: {total_return:.2f}%")

    best_thresh = max(results, key=results.get)
    print(f"\nBest Threshold: {best_thresh} with Return: {results[best_thresh]:.2f}%")

    # --- Visualization Run for Best Threshold ---
    print(f"\nGenerating visualization for Best Threshold: {best_thresh}...")
    thresh = best_thresh
    balance = 1000.0
    equity_curve = [balance]
    
    entry_indices = []
    entry_prices = []
    exit_indices = []
    exit_prices = []
    exit_colors = [] # Green for win, Red for loss
    
    in_trade = False
    entry_price = 0
    entry_idx = 0
    
    for i in range(len(df)):
        if in_trade:
            # Check exit conditions
            curr_low = df.loc[i, 'low']
            curr_high = df.loc[i, 'high']
            curr_close = df.loc[i, 'close']
            
            tp_price = entry_price * 1.01
            sl_price = entry_price * 0.99
            
            pnl = 0
            exit_reason = ""
            trade_done = False
            exec_price = curr_close # Default if time exit
            
            if curr_low <= sl_price:
                pnl = -0.01
                balance *= (1 + pnl)
                in_trade = False
                exit_reason = "SL"
                trade_done = True
                exec_price = sl_price
            elif curr_high >= tp_price:
                pnl = 0.01
                balance *= (1 + pnl)
                in_trade = False
                exit_reason = "TP"
                trade_done = True
                exec_price = tp_price
            elif i - entry_idx >= 32: # Time limit 8h
                pnl = (curr_close - entry_price) / entry_price
                balance *= (1 + pnl)
                in_trade = False
                exit_reason = "Time"
                trade_done = True
                exec_price = curr_close
            
            if trade_done:
                equity_curve.append(balance)
                exit_indices.append(i)
                exit_prices.append(exec_price)
                exit_colors.append('g' if pnl > 0 else 'r')
                
        elif df.loc[i, 'prob'] > thresh:
            # Enter Trade
            in_trade = True
            entry_price = df.loc[i, 'close']
            entry_idx = i
            
            entry_indices.append(i)
            entry_prices.append(entry_price)
    
    # Plot Trades
    plt.figure(figsize=(14, 7))
    # Plot price (subset or full?) Full might be messy but let's try
    plt.plot(df['close'].values, label='BTC Price', color='gray', alpha=0.5, linewidth=1)
    
    # Plot Entries
    plt.scatter(entry_indices, entry_prices, marker='^', color='blue', label='Buy', s=50, zorder=5)
    
    # Plot Exits
    # Split wins and losses for legend
    win_indices = [i for i, c in zip(exit_indices, exit_colors) if c == 'g']
    win_prices = [p for p, c in zip(exit_prices, exit_colors) if c == 'g']
    loss_indices = [i for i, c in zip(exit_indices, exit_colors) if c == 'r']
    loss_prices = [p for p, c in zip(exit_prices, exit_colors) if c == 'r']
    
    if win_indices:
        plt.scatter(win_indices, win_prices, marker='v', color='green', label='Win', s=50, zorder=5)
    if loss_indices:
        plt.scatter(loss_indices, loss_prices, marker='v', color='red', label='Loss', s=50, zorder=5)
        
    plt.title(f'Trades at Threshold {best_thresh} (Return: {results[best_thresh]:.2f}%)')
    plt.xlabel('Candles (15m)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('backtest_trades.png')
    print("Saved backtest_trades.png")
    
    # Plot Equity Curve
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve, color='purple', linewidth=2)
    plt.title('Account Balance Growth')
    plt.xlabel('Number of Trades')
    plt.ylabel('Balance ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig('backtest_balance.png')
    print("Saved backtest_balance.png")

if __name__ == "__main__":
    backtest_model()
