import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

def fetch_data(symbol='BTC/USDT', timeframe='15m', start_year=2017):
    """
    Fetches historical OHLCV data from Binance.
    
    Args:
        symbol (str): Trading pair symbol.
        timeframe (str): Timeframe interval.
        start_year (int): Year to start fetching data from.
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    print(f"Fetching {symbol} {timeframe} data starting from {start_year}...")
    exchange = ccxt.binance()
    
    # Calculate start time (milliseconds timestamp)
    start_date = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    since_ts = int(start_date.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000  # Binance limit
    
    # Current time
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update since_ts for next iteration (last timestamp + 1ms)
            last_ts = ohlcv[-1][0]
            since_ts = last_ts + 1
            
            # Break if we've reached current time
            if last_ts >= now_ts:
                break
                
            print(f"Fetched {len(ohlcv)} candles, last date: {datetime.fromtimestamp(last_ts/1000, timezone.utc)}")
            
            # Rate limit respect
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
            # Retry mechanism could be added here
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove potential duplicates
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    print(f"Total fetched: {len(df)} rows.")
    return df

if __name__ == "__main__":
    # Fetch data from 2017
    df = fetch_data(start_year=2017) 
    
    # Save to CSV
    output_path = 'data/btc_15m_2017_2025.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
