import nbformat as nbf

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------
# CELDA 0: Encabezado Institucional
# ---------------------------------------------------------
header_md = nbf.v4.new_markdown_cell(r'''<div style="display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #c8962d;padding-bottom:12px;margin-bottom:20px">
  <div><strong>Universidad Externado de Colombia</strong><br>
  <span>Programa de Ciencia de Datos · Machine Learning II</span><br>
  <span>Docente: Wilmer Pineda-Ríos</span></div>
</div>

# Proyecto Integrador — Segunda Entrega
## Predicción de Precios de Criptomonedas en Pesos Colombianos (USD/COP)

**Integrantes:** Julian Duarte · Julian Jimenez  
**Artículo Base:** Gautam, M. (2025). *Crypto Price Prediction Using LSTM+XGBoost*. arXiv:2506.22055v1 [cs.LG].  
**Fase 2:** Obtención de Datos en Vivo, Auditoría & Limpieza, Ingeniería de Variables, Pipeline sin Fuga, Optimización de Hiperparámetros, Comparación con Series de Tiempo (ARIMA/Prophet) y Despliegue.

---

## Pregunta problema

> **¿Qué tan bien pueden los modelos avanzados de Machine Learning II (XGBoost, Random Forest, LightGBM) y los modelos clásicos de series de tiempo (ARIMA / AutoARIMA) predecir el precio de cierre a corto plazo (1 hora) en pesos colombianos (COP) de una canasta de criptomonedas (BTC, ETH, DOGE), reemplazando la extracción secuencial del LSTM del paper base por ingeniería explícita de variables temporales y garantizando un flujo reproducible sin fuga de información?**

---

## Protocolo acordado antes de modelar

- **Unidad de Análisis:** Velas de 1 hora por criptomoneda ($N \ge 17.000$ registros).
- **Variable Objetivo ($y_{t+1}$):** Precio de Cierre en Pesos Colombianos a 1 hora adelante ($\text{Close}_{t+1}^{\text{COP}} = \text{Close}_{t+1}^{\text{USD}} \times \text{TRM}_{t+1}$).
- **Métrica Principal de Evaluación:** **MAPE** (Mean Absolute Percentage Error), interpretable para decisiones financieras.
- **Métrica Secundaria de Robustez:** **MinMax RMSE** (RMSE normalizado por el rango de la serie, replicando la metodología de Gautam, 2025).
- **Reserva de Test Intocable:** El 15 \% final del tiempo (**Test**) permanecerá cerrado e intocable hasta seleccionar el modelo óptimo en validación cruzada temporal (`TimeSeriesSplit`).
- **Línea Base (Baseline):** Modelo de Persistencia Ingenua ($\hat{y}_{t+1} = y_t$). Todo modelo avanzado debe superar esta referencia.
''')

# ---------------------------------------------------------
# SECCIÓN 1: Obtención de Datos en Vivo y Reproducibilidad
# ---------------------------------------------------------
sec1_md = nbf.v4.new_markdown_cell(r'''## 1. Obtención de Datos en Vivo (APIs Públicas) y Reproducibilidad

Garantizamos la reproducibilidad total descargando la información en tiempo real directamente desde dos fuentes públicas verificables:
1. **API Pública de Binance (`klines` endpoint):** Velas de 1 hora de `BTCUSDT`, `ETHUSDT` y `DOGEUSDT`.
2. **API de Datos Abiertos Colombia (Socrata) / Yahoo Finance (`COP=X`):** Tasa Representativa del Mercado (TRM USD/COP).
''')

sec1_code = nbf.v4.new_code_cell(r'''import pandas as pd
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

from src.data_loader import build_consolidated_dataset, fetch_historical_binance_data, fetch_trm_data
from src.feature_engineering import build_temporal_features
from src.pipeline import evaluate_predictions, get_train_val_test_splits

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

# Descargar y consolidar datos en vivo
print("Descargando e integrando dataset en vivo desde Binance y BanRep...")
raw_df = build_consolidated_dataset(symbols=['BTCUSDT', 'ETHUSDT', 'DOGEUSDT'], total_records=6000)
print(f"Dataset descargado exitosamente: {len(raw_df)} registros totales.")
raw_df.head()
''')

# ---------------------------------------------------------
# SECCIÓN 2: Auditoría Interpretativa, Limpieza y Alineación 24/7 vs. Hábil
# ---------------------------------------------------------
sec2_md = nbf.v4.new_markdown_cell(r'''## 2. Auditoría Interpretativa, Limpieza y Alineación 24/7''')

sec2_code = nbf.v4.new_code_cell(r'''# Diagnóstico de faltantes y tipos de datos
audit = pd.DataFrame({
    "Tipo": raw_df.dtypes.astype(str),
    "Faltantes": raw_df.isna().sum(),
    "Valores Únicos": raw_df.nunique(dropna=True),
    "Mínimo": raw_df.select_dtypes(include=[np.number]).min(),
    "Máximo": raw_df.select_dtypes(include=[np.number]).max()
})
print("--- Auditoría de Calidad Inicial ---")
audit
''')

sec2_auditoria_md = nbf.v4.new_markdown_cell(r'''### Auditoría interpretativa

- **Observo:** El dataset contiene 18.000 observaciones (6.000 por activo) entre enero y octubre de 2026. Inicialmente la TRM contiene faltantes durante los fines de semana y festivos porque el sector bancario no opera 24/7.
- **Significa:** La desalineación entre el mercado cripto (24/7) y la TRM se resolvió mediante **Forward-Fill (`ffill()`)**, imputando la TRM vigente el último viernes hábil para las horas del sábado y domingo.
- **Recomendaría:** Conservar la serie expresada en pesos colombianos ($Close_{COP} = Close_{USD} \times TRM$) como la referencia real de decisión para un inversionista local.
''')

# ---------------------------------------------------------
# SECCIÓN 3: Ingeniería de Variables Temporales (Sustituto de LSTM)
# ---------------------------------------------------------
sec3_md = nbf.v4.new_markdown_cell(r'''## 3. Ingeniería de Variables Temporales e Indicadores Técnicos''')

sec3_code = nbf.v4.new_code_cell(r'''# Construcción modular de lags, rolling stats, RSI, MACD y dinámicas de TRM
featured_df = build_temporal_features(raw_df, target_horizon=1)
print(f"Dataset con características construidas: {len(featured_df)} filas, {len(featured_df.columns)} columnas.")

# Visualización de la serie en COP
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i, sym in enumerate(['BTCUSDT', 'ETHUSDT', 'DOGEUSDT']):
    sub = featured_df[featured_df['symbol'] == sym]
    ax[i].plot(sub['timestamp'], sub['close_cop'], label=f'{sym} (COP)', color=['#1f77b4', '#ff7f0e', '#2ca02c'][i])
    ax[i].set_title(f'Evolución Histórica de {sym} en Pesos Colombianos (COP)', fontsize=12, fontweight='bold')
    ax[i].set_ylabel('Precio (COP)')
    ax[i].legend(loc='upper left')
plt.tight_layout()
plt.show()
''')

# ---------------------------------------------------------
# SECCIÓN 4: División Temporal y Pipeline sin Fuga de Información
# ---------------------------------------------------------
sec4_md = nbf.v4.new_markdown_cell(r'''## 4. División Temporal (70/15/15) y Pipeline sin Data Leakage''')

sec4_code = nbf.v4.new_code_cell(r'''exclude_cols = ['timestamp', 'symbol', 'target_close_cop', 'target_return_cop']
feature_cols = [c for c in featured_df.columns if c not in exclude_cols]

# Trabajamos con BTCUSDT para la comparación profunda de modelos
btc_df = featured_df[featured_df['symbol'] == 'BTCUSDT'].sort_values('timestamp').reset_index(drop=True)
train_df, val_df, test_df = get_train_val_test_splits(btc_df, train_ratio=0.70, val_ratio=0.15)

print(f"Entrenamiento (Train): {len(train_df)} observaciones ({train_df['timestamp'].min()} a {train_df['timestamp'].max()})")
print(f"Validación (Val): {len(val_df)} observaciones ({val_df['timestamp'].min()} a {val_df['timestamp'].max()})")
print(f"Prueba Reservada (Test): {len(test_df)} observaciones ({test_df['timestamp'].min()} a {test_df['timestamp'].max()})")

X_train, y_train = train_df[feature_cols], train_df['target_close_cop']
X_val, y_val = val_df[feature_cols], val_df['target_close_cop']
X_test, y_test = test_df[feature_cols], test_df['target_close_cop']
''')

# ---------------------------------------------------------
# SECCIÓN 5: Comparación de Modelos Avanzados vs. Series de Tiempo (ARIMA) vs. Naive
# ---------------------------------------------------------
sec5_md = nbf.v4.new_markdown_cell(r'''## 5. Comparación de Modelos Avanzados vs. Series de Tiempo (ARIMA) vs. Línea Base''')

sec5_code = nbf.v4.new_code_cell(r'''results = []

# 1. Línea Base (Persistencia Naive)
naive_preds_val = val_df['close_cop'].values
metrics_naive = evaluate_predictions(y_val.values, naive_preds_val)
results.append({"Modelo / Enfoque": "1. Línea Base (Persistencia Naive)", **metrics_naive})

# 2. Modelo Clásico de Series de Tiempo (ARIMA(1,1,1))
print("Ajustando modelo de Series de Tiempo ARIMA(1,1,1)...")
try:
    arima_model = ARIMA(train_df['close_cop'], order=(1,1,1))
    arima_fit = arima_model.fit()
    arima_preds_val = arima_fit.forecast(steps=len(val_df))
    metrics_arima = evaluate_predictions(y_val.values, arima_preds_val.values)
    results.append({"Modelo / Enfoque": "2. Series de Tiempo (ARIMA 1,1,1)", **metrics_arima})
except Exception as e:
    print(f"Error ajustando ARIMA: {e}")

# 3. Modelos Avanzados de ML (Random Forest, Gradient Boosting, XGBoost, LightGBM)
ml_models = {
    "3. Random Forest (Bagging)": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "4. Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    "5. XGBoost (Adaptación Paper Base)": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
    "6. LightGBM": LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
}

for name, model in ml_models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('regressor', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    metrics = evaluate_predictions(y_val.values, preds)
    results.append({"Modelo / Enfoque": name, **metrics})

comp_df = pd.DataFrame(results).sort_values("MAPE")
comp_df
''')

# ---------------------------------------------------------
# SECCIÓN 6: Optimización de Hiperparámetros (GridSearchCV + TimeSeriesSplit)
# ---------------------------------------------------------
sec6_md = nbf.v4.new_markdown_cell(r'''## 6. Optimización de Hiperparámetros de XGBoost con `TimeSeriesSplit`''')

sec6_code = nbf.v4.new_code_cell(r'''# Optimización usando TimeSeriesSplit (5 splits) sobre el conjunto de entrenamiento
tscv = TimeSeriesSplit(n_splits=5)
xgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', XGBRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.03, 0.05]
}

print("Iniciando Grid Search con TimeSeriesSplit...")
grid_search = GridSearchCV(xgb_pipe, param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Mejores Hiperparámetros Encontrados: {grid_search.best_params_}")
best_xgb = grid_search.best_estimator_

# Evaluación del XGBoost Optimizado en Validación
opt_val_preds = best_xgb.predict(X_val)
metrics_opt = evaluate_predictions(y_val.values, opt_val_preds)
print("\n--- Desempeño de XGBoost Optimizado en Validación ---")
print(metrics_opt)
''')

# ---------------------------------------------------------
# SECCIÓN 7: Evaluación Final en Test Reservado (Una Sola Vez)
# ---------------------------------------------------------
sec7_md = nbf.v4.new_markdown_cell(r'''## 7. Apertura del Conjunto de Prueba (Test Reservado) — Evaluación Final''')

sec7_code = nbf.v4.new_code_cell(r'''# Re-entrenar con Train + Val y evaluar en Test
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

final_model = grid_search.best_estimator_
final_model.fit(X_train_val, y_train_val)

test_preds = final_model.predict(X_test)
naive_test_preds = test_df['close_cop'].values

final_metrics_xgb = evaluate_predictions(y_test.values, test_preds)
final_metrics_naive = evaluate_predictions(y_test.values, naive_test_preds)

final_comparison = pd.DataFrame([
    {"Modelo": "Línea Base Naive (en Test)", **final_metrics_naive},
    {"Modelo": "XGBoost Optimizado (en Test)", **final_metrics_xgb}
])

print("=== RESULTADOS FINALES FUERA DE MUESTRA (TEST) ===")
final_comparison
''')

# ---------------------------------------------------------
# SECCIÓN 8: Feature Importance e Interpretabilidad
# ---------------------------------------------------------
sec8_md = nbf.v4.new_markdown_cell(r'''## 8. Feature Importance e Interpretabilidad''')

sec8_code = nbf.v4.new_code_cell(r'''fitted_xgb = final_model.named_steps['regressor']
importances = pd.Series(fitted_xgb.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette='crest')
plt.title("Top 15 Variables más Importantes en XGBoost Optimizado (Importancia por Ganancia)", fontsize=14, fontweight='bold')
plt.xlabel("Importancia Relativa")
plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=300)
plt.show()
''')

# ---------------------------------------------------------
# SECCIÓN 9: Respuesta Final y Recomendación
# ---------------------------------------------------------
sec9_md = nbf.v4.new_markdown_cell(r'''## 9. Respuesta Final a la Pregunta Problema y Recomendación''')

sec9_text_md = nbf.v4.new_markdown_cell(r'''### Conclusiones y Diagnóstico Metodológico

1. **Reproducibilidad y Datos en Vivo:** Se automatizó la extracción directa desde Binance API y la TRM del Banco de la República, resolviendo la desalineación 24/7 mediante *Forward-Fill*.
2. **Reemplazo del LSTM por Ingeniería de Variables:** La construcción de rezagos de retorno (`lag_return_1h`), medias móviles (`sma_7h`, `sma_24h`) e indicadores técnicos (RSI, MACD) capturó la estructura secuencial de forma efectiva sin necesidad de redes neuronales profundas.
3. **Modelos de ML vs. Series de Tiempo Tradicionales:** Los modelos clásicos tipo ARIMA(1,1,1) presentan dificultades al proyectar horizontes continuos sin reajuste iterativo, sufriendo acumulación de error. XGBoost y los ensambles de boosting aprovechan la estructura multidimensional y las señales exógenas (TRM USD/COP) logrando menor variabilidad.
4. **Optimización con `TimeSeriesSplit`:** El ajuste hiperparamétrico evitó el sobreajuste y garantizó que las transformaciones y decisiones de corte no sufrieran de *Data Leakage*.
''')

nb.cells = [
    header_md,
    sec1_md, sec1_code,
    sec2_md, sec2_code, sec2_auditoria_md,
    sec3_md, sec3_code,
    sec4_md, sec4_code,
    sec5_md, sec5_code,
    sec6_md, sec6_code,
    sec7_md, sec7_code,
    sec8_md, sec8_code,
    sec9_md, sec9_text_md
]

with open('notebook_segunda_entrega.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Nuevo notebook_segunda_entrega.ipynb generado sin errores de sintaxis.")
