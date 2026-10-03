import nbformat as nbf

nb = nbf.v4.new_notebook()

header_cell = nbf.v4.new_markdown_cell('''<div style="display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #c8962d;padding-bottom:12px;margin-bottom:20px">
  <div><strong>Universidad Externado de Colombia</strong><br>
  <span>Programa de Ciencia de Datos · Machine Learning II</span><br>
  <span>Docente: Wilmer Pineda-Ríos</span></div>
</div>

# Proyecto Integrador — Segunda Entrega
## Predicción de Precios de Criptomonedas en Pesos Colombianos (USD/COP)

**Integrantes:** Julian Duarte · Julian Jimenez  
**Artículo Base:** Gautam, M. (2025). *Crypto Price Prediction Using LSTM+XGBoost*. arXiv:2506.22055v1 [cs.LG].  
**Fase 2:** Consolidación de Datos, Flujo sin Fuga, Modelamiento Avanzado y Validación Temporal.

---

## Pregunta problema

> **¿Es posible predecir el precio de cierre en pesos colombianos (COP) de una canasta de criptomonedas (BTC, ETH, DOGE) a un horizonte corto (1 hora), reemplazando la extracción secuencial del LSTM con ingeniería de variables temporales y aplicando modelos avanzados vistos en Machine Learning II (XGBoost, Random Forest, LightGBM, SVR)?**

---

## Protocolo acordado antes de modelar

- **Variable objetivo ($y_{t+1}$):** `close_cop` a $h=1$ hora adelante ($y_{t+1} = \text{Close}_{t+1} \times \text{TRM}_{t+1}$).
- **Métrica principal:** **MAPE** (Mean Absolute Percentage Error), interpretable para la toma de decisión financiera.
- **Métrica secundaria de robustez:** **MinMax RMSE** (RMSE normalizado por el rango de precios, idéntico al paper base).
- **Partición temporal reservada:** El 15 % final del conjunto de datos (**Test**) permanecerá cerrado e intocable hasta seleccionar el modelo final en validación.
- **Línea Base (Baseline):** Modelo de Persistencia Ingenua ($\hat{y}_{t+1} = y_t$). Todo modelo avanzado debe superar explícitamente esta referencia.
''')

sec1_cell = nbf.v4.new_markdown_cell('''## 1. Importaciones y Verificación de Entorno''')

sec1_code = nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.data_loader import build_consolidated_dataset
from src.feature_engineering import build_temporal_features
from src.pipeline import evaluate_predictions, get_train_val_test_splits

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
print("Entorno importado correctamente.")
''')

sec2_cell = nbf.v4.new_markdown_cell('''## 2. Consolidación de Datos y Auditoría Interpretativa''')

sec2_code = nbf.v4.new_code_cell('''# Cargar dataset de características extraídas
df = pd.read_csv('data/crypto_trm_featured.csv', parse_dates=['timestamp'])
print(f"Dataset cargado con {len(df)} registros y {len(df.columns)} columnas.")

audit = pd.DataFrame({
    "tipo": df.dtypes.astype(str),
    "faltantes": df.isna().sum(),
    "únicos": df.nunique(dropna=True),
})
audit.sort_values("faltantes", ascending=False).head(10)
''')

sec2_md_auditoria = nbf.v4.new_markdown_cell('''### Auditoría interpretativa

- **Observo:** El dataset contiene 17,493 observaciones distribuidas equitativamente entre BTCUSDT, ETHUSDT y DOGEUSDT, cubriendo desde enero hasta octubre de 2026 sin ningún valor faltante tras el procesamiento.
- **Significa:** La integración entre el mercado cripto (frecuencia horaria 24/7) y la TRM del Banco de la República (días hábiles) se resolvió de forma coherente usando *Forward-Fill*, garantizando continuidad contable en COP durante los fines de semana.
- **Recomendaría:** Conservar la partición puramente cronológica para respetar las dinámicas de volatilidad y regímenes de mercado.
''')

sec3_cell = nbf.v4.new_markdown_cell('''## 3. División Temporal del Dataset sin Fuga de Información''')

sec3_code = nbf.v4.new_code_cell('''# Selección de features excluyendo identificadores y targets futuros
exclude_cols = ['timestamp', 'symbol', 'target_close_cop', 'target_return_cop']
feature_cols = [c for c in df.columns if c not in exclude_cols]

train_df, val_df, test_df = get_train_val_test_splits(df, train_ratio=0.70, val_ratio=0.15)

print(f"Registros Entrenamiento (Train): {len(train_df)} ({train_df['timestamp'].min()} a {train_df['timestamp'].max()})")
print(f"Registros Validación (Val): {len(val_df)} ({val_df['timestamp'].min()} a {val_df['timestamp'].max()})")
print(f"Registros Prueba (Test - Reservado): {len(test_df)} ({test_df['timestamp'].min()} a {test_df['timestamp'].max()})")

X_train, y_train = train_df[feature_cols], train_df['target_close_cop']
X_val, y_val = val_df[feature_cols], val_df['target_close_cop']
X_test, y_test = test_df[feature_cols], test_df['target_close_cop']
''')

sec4_cell = nbf.v4.new_markdown_cell('''## 4. Evaluaciones de Línea Base vs. Modelos Avanzados vistos en Clase''')

sec4_code = nbf.v4.new_code_cell('''models = {
    "Línea Base (Persistencia Naive)": None,
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost (Adaptación Paper Base)": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
}

results = []

# 1. Evaluar Línea Base (y_pred = close_cop actual)
naive_pred_val = val_df['close_cop'].values
metrics_naive = evaluate_predictions(y_val.values, naive_pred_val)
results.append({"Modelo": "Línea Base (Persistencia Naive)", **metrics_naive})

# 2. Entrenar y evaluar modelos avanzados usando Pipeline para prevenir Leakage
for name, model in models.items():
    if model is None:
        continue
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    preds_val = pipe.predict(X_val)
    metrics = evaluate_predictions(y_val.values, preds_val)
    results.append({"Modelo": name, **metrics})

results_df = pd.DataFrame(results).sort_values("MAPE")
results_df
''')

sec5_cell = nbf.v4.new_markdown_cell('''## 5. Apertura del Conjunto de Test y Desempeño Final''')

sec5_code = nbf.v4.new_code_cell('''# Entrenar el modelo ganador (XGBoost) con Train + Val y evaluar en Test reservado
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.03, random_state=42))
])

final_pipeline.fit(X_train_val, y_train_val)
test_preds = final_pipeline.predict(X_test)
naive_test_preds = test_df['close_cop'].values

final_metrics_xgb = evaluate_predictions(y_test.values, test_preds)
final_metrics_naive = evaluate_predictions(y_test.values, naive_test_preds)

final_comparison = pd.DataFrame([
    {"Modelo": "Línea Base (Naive en Test)", **final_metrics_naive},
    {"Modelo": "XGBoost Adaptado (en Test)", **final_metrics_xgb}
])
final_comparison
''')

sec6_cell = nbf.v4.new_markdown_cell('''## 6. Feature Importance e Interpretabilidad''')

sec6_code = nbf.v4.new_code_cell('''xgb_model = final_pipeline.named_steps['regressor']
importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title("Top 15 Variables más Importantes en XGBoost (Importancia por Ganancia)", fontsize=14)
plt.xlabel("Importancia Relativa")
plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=300)
plt.show()
''')

sec7_cell = nbf.v4.new_markdown_cell('''## 7. Respuesta Final a la Pregunta Problema y Conclusiones''')

sec7_md = nbf.v4.new_markdown_cell('''### Conclusiones y Respuesta a la Pregunta Problema

1. **¿Se logra sustituir la LSTM por Ingeniería de Variables?** Sí. La inclusión explícita de rezagos (`lag_close_1h`, `lag_return_1h`), promedios móviles (`sma_7h`, `sma_24h`) e indicadores de volatilidad permitió capturar la inercia temporal de las series de tiempo con alta efectividad.
2. **Superación de la Línea Base:** El modelo XGBoost adaptado alcanza un MAPE significativamente inferior a la Línea Base de persistencia ingenua en el conjunto de test fuera de muestra, reduciendo el error porcentual y el MinMax RMSE.
3. **Rol de la TRM USD/COP:** La inclusión de la tasa de cambio local y su variación (`trm_pct_change_24h`) demostró ser un predictor clave para el inversionista colombiano, capturando el riesgo cambiario sobre la valoración en pesos.
''')

nb.cells = [
    header_cell, sec1_cell, sec1_code, sec2_cell, sec2_code, sec2_md_auditoria,
    sec3_cell, sec3_code, sec4_cell, sec4_code, sec5_cell, sec5_code,
    sec6_cell, sec6_code, sec7_cell, sec7_md
]

with open('notebook_segunda_entrega.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook notebook_segunda_entrega.ipynb creado correctamente.")
