import pandas as pd
import numpy as np
import urllib.request
import json
import yfinance as yf
from datetime import datetime, timedelta

def fetch_historical_binance_data(symbol='BTCUSDT', interval='1h', total_records=6000):
    """
    Descarga iterativamente velas históricas desde la API pública de Binance.
    """
    all_dfs = []
    end_time = None
    records_fetched = 0
    
    while records_fetched < total_records:
        limit = min(1000, total_records - records_fetched)
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if end_time:
            url += f"&endTime={end_time}"
            
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                if not data:
                    break
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df['symbol'] = symbol
                df_clean = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                
                all_dfs.append(df_clean)
                records_fetched += len(df_clean)
                end_time = int(data[0][0]) - 1  # retroceder en el tiempo
        except Exception as e:
            print(f"Error en iteración Binance para {symbol}: {e}")
            break
            
    if all_dfs:
        full_df = pd.concat(all_dfs).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return full_df
    return pd.DataFrame()

def fetch_trm_data():
    """
    Descarga serie histórica de TRM USD/COP usando Yahoo Finance (`COP=X`) o API secundaria de Datos Abiertos.
    """
    try:
        # Intento 1: API Socrata Datos Abiertos Colombia (Endpoint v2)
        url = "https://www.datos.gov.co/resource/32sa-823r.json?$limit=5000"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if isinstance(data, list) and len(data) > 0 and 'valor' in data[0]:
                df = pd.DataFrame(data)
                df['fecha'] = pd.to_datetime(df.get('vigenciadesde', df.get('fecha')))
                df['trm'] = df['valor'].astype(float)
                df = df[['fecha', 'trm']].sort_values('fecha').reset_index(drop=True)
                return df
    except Exception as e:
        print(f"Socrata fallback a Yahoo Finance por: {e}")
        
    # Intento 2: yfinance COP=X
    try:
        cop = yf.Ticker("COP=X")
        df_hist = cop.history(period="2y", interval="1d").reset_index()
        df_hist['fecha'] = pd.to_datetime(df_hist['Date']).dt.tz_localize(None)
        df_hist['trm'] = df_hist['Close'].astype(float)
        return df_hist[['fecha', 'trm']].sort_values('fecha').reset_index(drop=True)
    except Exception as e:
        print(f"Error descargando TRM con yfinance: {e}")
        return pd.DataFrame()

def build_consolidated_dataset(symbols=['BTCUSDT', 'ETHUSDT', 'DOGEUSDT'], total_records=6000):
    """
    Consolida precios horarios de criptos con TRM diaria mediante Forward Fill.
    """
    trm_df = fetch_trm_data()
    if trm_df.empty:
        raise ValueError("No se pudo obtener la serie de la TRM USD/COP")
        
    crypto_dfs = []
    for sym in symbols:
        print(f"Descargando {sym}...")
        df_sym = fetch_historical_binance_data(sym, interval='1h', total_records=total_records)
        crypto_dfs.append(df_sym)
        
    full_crypto = pd.concat(crypto_dfs).reset_index(drop=True)
    full_crypto['date_only'] = full_crypto['timestamp'].dt.floor('d')
    trm_df['date_only'] = trm_df['fecha'].dt.floor('d')
    
    # Merge y forward-fill para alinear 24/7 con TRM bancaria
    merged = pd.merge(full_crypto, trm_df[['date_only', 'trm']], on='date_only', how='left')
    merged['trm'] = merged['trm'].ffill().bfill()
    
    # Calcular precio en COP
    merged['close_cop'] = merged['close'] * merged['trm']
    merged['open_cop'] = merged['open'] * merged['trm']
    merged['high_cop'] = merged['high'] * merged['trm']
    merged['low_cop'] = merged['low'] * merged['trm']
    
    return merged.drop(columns=['date_only'])

if __name__ == '__main__':
    print("Consolidando dataset real (Binance + TRM USD/COP)...")
    dataset = build_consolidated_dataset(symbols=['BTCUSDT', 'ETHUSDT', 'DOGEUSDT'], total_records=6000)
    print(f"Dataset consolidado exitosamente: {len(dataset)} filas, {len(dataset.columns)} columnas.")
    print(dataset.head())
    dataset.to_csv("data/crypto_trm_dataset.csv", index=False)
    print("Guardado en data/crypto_trm_dataset.csv")
