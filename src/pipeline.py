import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

def min_max_rmse(y_true, y_pred):
    """
    MinMax RMSE tal como se define en el artículo de Gautam (2025):
    Normaliza el RMSE dividiéndolo entre el rango (max - min) de la variable observada.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    val_range = np.max(y_true) - np.min(y_true)
    return rmse / (val_range + 1e-8)

def evaluate_predictions(y_true, y_pred):
    """
    Retorna métricas clave: MAPE, RMSE en COP y MinMax RMSE.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse_cop = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mm_rmse = min_max_rmse(y_true, y_pred)
    return {
        'MAPE': mape,
        'RMSE_COP': rmse_cop,
        'MinMax_RMSE': mm_rmse
    }

class TimeSeriesFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selector de columnas numéricas para modelos de regresión temporal.
    """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].values
        return X

def get_train_val_test_splits(df, train_ratio=0.70, val_ratio=0.15):
    """
    Partición cronológica sin desordenar (sin shuffle) para evitar Data Leakage temporal.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df
