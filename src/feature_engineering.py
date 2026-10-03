import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    """Calcula el Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def build_temporal_features(df, target_horizon=1):
    """
    Construye variables temporales, rezagadas e indicadores técnicos por criptomoneda
    para evitar contaminación entre símbolos.
    """
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    featured_dfs = []
    
    for sym, group in df.groupby('symbol'):
        g = group.copy()
        
        # 1. Retorno porcentual horario en USD y COP
        g['return_1h'] = g['close'].pct_change()
        g['return_cop_1h'] = g['close_cop'].pct_change()
        
        # 2. Variable Objetivo: Precio de cierre en COP a h horas adelante
        g['target_close_cop'] = g['close_cop'].shift(-target_horizon)
        g['target_return_cop'] = g['close_cop'].pct_change(target_horizon).shift(-target_horizon)
        
        # 3. Lags de Precio y Retorno (USD y COP)
        for lag in [1, 2, 3, 6, 12, 24]:
            g[f'lag_close_{lag}h'] = g['close'].shift(lag)
            g[f'lag_return_{lag}h'] = g['return_1h'].shift(lag)
            g[f'lag_trm_{lag}h'] = g['trm'].shift(lag)
            
        # 4. Estadísticas Móviles (Rolling)
        for window in [7, 14, 24, 168]: # 7h, 14h, 24h (1 día), 168h (1 semana)
            g[f'sma_{window}h'] = g['close'].rolling(window=window).mean()
            g[f'volatility_{window}h'] = g['return_1h'].rolling(window=window).std()
            g[f'close_ratio_sma_{window}h'] = g['close'] / (g[f'sma_{window}h'] + 1e-8)
            
        # 5. Indicadores Técnicos
        g['rsi_14'] = compute_rsi(g['close'], period=14)
        ema_12 = g['close'].ewm(span=12, adjust=False).mean()
        ema_26 = g['close'].ewm(span=26, adjust=False).mean()
        g['macd'] = ema_12 - ema_26
        g['macd_signal'] = g['macd'].ewm(span=9, adjust=False).mean()
        g['high_low_ratio'] = (g['high'] - g['low']) / (g['low'] + 1e-8)
        
        # 6. Dinámica de TRM (Macro exógena)
        g['trm_pct_change_24h'] = g['trm'].pct_change(24)
        
        featured_dfs.append(g)
        
    final_df = pd.concat(featured_dfs).reset_index(drop=True)
    # Eliminar filas iniciales/finales con NAs debido a lags/rolling/shift
    final_df = final_df.dropna(subset=['target_close_cop', 'rsi_14', 'volatility_168h']).reset_index(drop=True)
    return final_df

if __name__ == '__main__':
    print("Ejecutando ingeniería de variables temporales...")
    raw_df = pd.read_csv('data/crypto_trm_dataset.csv', parse_dates=['timestamp'])
    feat_df = build_temporal_features(raw_df, target_horizon=1)
    print(f"Dataset con variables temporales: {len(feat_df)} observaciones, {len(feat_df.columns)} columnas.")
    feat_df.to_csv("data/crypto_trm_featured.csv", index=False)
    print("Guardado exitosamente en data/crypto_trm_featured.csv")
