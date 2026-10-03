import pandas as pd
import numpy as np
import urllib.request
import json
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA

from src.data_loader import build_consolidated_dataset
from src.feature_engineering import build_temporal_features
from src.pipeline import evaluate_predictions, get_train_val_test_splits

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

print("--- 1. Carga e Integración de Datos ---")
raw_df = pd.read_csv('data/crypto_trm_dataset.csv')
featured_df = pd.read_csv('data/crypto_trm_featured.csv', parse_dates=['timestamp'])

exclude_cols = ['timestamp', 'symbol', 'target_close_cop', 'target_return_cop']
feature_cols = [c for c in featured_df.columns if c not in exclude_cols]

btc_df = featured_df[featured_df['symbol'] == 'BTCUSDT'].sort_values('timestamp').reset_index(drop=True)
train_df, val_df, test_df = get_train_val_test_splits(btc_df, train_ratio=0.70, val_ratio=0.15)

X_train, y_train = train_df[feature_cols], train_df['target_close_cop']
X_val, y_val = val_df[feature_cols], val_df['target_close_cop']
X_test, y_test = test_df[feature_cols], test_df['target_close_cop']

print("--- 2. Evaluación de Modelos en Validación ---")
results = []
metrics_naive = evaluate_predictions(y_val.values, val_df['close_cop'].values)
results.append({"Modelo / Enfoque": "1. Línea Base (Persistencia Naive)", **metrics_naive})

arima_fit = ARIMA(train_df['close_cop'], order=(1,1,1)).fit()
arima_preds = arima_fit.forecast(steps=len(val_df))
metrics_arima = evaluate_predictions(y_val.values, arima_preds.values)
results.append({"Modelo / Enfoque": "2. Series de Tiempo (ARIMA 1,1,1)", **metrics_arima})

models = {
    "3. Random Forest": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    "4. Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
    "5. XGBoost (Paper Base)": XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.03, random_state=42),
    "6. LightGBM": LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.03, random_state=42, verbose=-1)
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    results.append({"Modelo / Enfoque": name, **evaluate_predictions(y_val.values, preds)})

comp_df = pd.DataFrame(results).sort_values("MAPE")
print(comp_df)

print("\n--- 3. GridSearch XGBoost con TimeSeriesSplit ---")
tscv = TimeSeriesSplit(n_splits=3)
xgb_pipe = Pipeline([('scaler', StandardScaler()), ('regressor', XGBRegressor(random_state=42))])
param_grid = {'regressor__n_estimators': [30, 50], 'regressor__max_depth': [3, 5], 'regressor__learning_rate': [0.03, 0.05]}

grid = GridSearchCV(xgb_pipe, param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
grid.fit(X_train, y_train)
print("Mejores parámetros:", grid.best_params_)

print("\n--- 4. Evaluación en Test Reservado ---")
X_tr_v = pd.concat([X_train, X_val])
y_tr_v = pd.concat([y_train, y_val])
best_xgb = grid.best_estimator_
best_xgb.fit(X_tr_v, y_tr_v)

test_preds = best_xgb.predict(X_test)
naive_test = test_df['close_cop'].values

final_comp = pd.DataFrame([
    {"Modelo": "Línea Base Naive (en Test)", **evaluate_predictions(y_test.values, naive_test)},
    {"Modelo": "XGBoost Optimizado (en Test)", **evaluate_predictions(y_test.values, test_preds)}
])
print(final_comp)
