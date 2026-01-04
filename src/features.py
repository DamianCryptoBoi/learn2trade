import pandas as pd
import pandas_ta as ta
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def add_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    """
    # Custom Strategy
    # RSI
    df['RSI'] = df.ta.rsi(length=14)
    
    # MACD
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # ADX
    adx = df.ta.adx(length=14)
    df = pd.concat([df, adx], axis=1)
    
    # Bollinger Bands
    bb = df.ta.bbands(length=20, std=2)
    df = pd.concat([df, bb], axis=1)
    
    # ATR
    df['ATR'] = df.ta.atr(length=14)
    
    # OBV
    df['OBV'] = df.ta.obv()
    
    # SMA/EMA
    df['SMA_50'] = df.ta.sma(length=50)
    df['EMA_20'] = df.ta.ema(length=20)
    
    # Stochastic
    stoch = df.ta.stoch()
    df = pd.concat([df, stoch], axis=1)
    
    return df

def create_target(df, window=32):
    """
    Creates target label:
    1 if price hits TP (+1.0%) before SL (-1.0%) within window (8 hours = 32 candles).
    0 otherwise.
    """
    targets = []
    
    # Convert to numpy for speed
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    n = len(df)
    
    for i in range(n):
        if i + 1 >= n:
            targets.append(0)
            continue
            
        # Look forward
        end_idx = min(i + window, n)
        
        future_highs = high[i+1:end_idx]
        future_lows = low[i+1:end_idx]
        
        entry_price = close[i]
        tp_price = entry_price * 1.01
        sl_price = entry_price * 0.99
        
        outcome = 0 # Loss/Neutral
        
        for j in range(len(future_highs)):
            curr_high = future_highs[j]
            curr_low = future_lows[j]
            
            if curr_low <= sl_price:
                # Hit SL first (or same candle)
                outcome = 0
                break
            if curr_high >= tp_price:
                # Hit TP
                outcome = 1
                break
        
        targets.append(outcome)
        
    df['target'] = targets
    return df

def preprocess_data(input_path, output_path):
    print("Loading data...")
    df = load_data(input_path)
    
    print("Adding indicators...")
    df = add_indicators(df)
    
    print("Creating targets...")
    df = create_target(df)
    
    # Drop rows with NaN (due to indicators)
    df.dropna(inplace=True)
    
    # Split by Year
    # Train: 2017-2024
    # Test: 2025-
    
    train_df = df[(df['datetime'].dt.year >= 2017) & (df['datetime'].dt.year <= 2024)]
    test_df = df[df['datetime'].dt.year >= 2025]
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Train positive rate: {train_df['target'].mean():.4f}")
    print(f"Test positive rate: {test_df['target'].mean():.4f}")
    
    # Save processed data
    # We might want to normalize here or inside the Dataset class. 
    # Usually better to compute scaler on Train and apply to Test.
    # For now, let's save the raw features and target.
    
    train_path = output_path.replace('.csv', '_train.csv')
    test_path = output_path.replace('.csv', '_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")

if __name__ == "__main__":
    input_csv = 'data/btc_15m_2017_2025.csv'
    output_base = 'data/btc_features.csv'
    preprocess_data(input_csv, output_base)
