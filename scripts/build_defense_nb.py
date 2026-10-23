"""
Script para generar el notebook maestro de preparación de la sustentación oral
usando únicamente la librería estándar de Python (json).
"""

import json

header_md = r'''<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
  <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.25); padding-bottom: 12px; margin-bottom: 15px;">
    <div>
      <span style="font-size: 1.1em; letter-spacing: 1px; font-weight: 300;">UNIVERSIDAD EXTERNADO DE COLOMBIA</span><br>
      <strong style="font-size: 1.3em;">Maestría en Inteligencia de Negocios & Ciencia de Datos</strong>
    </div>
    <div style="text-align: right;">
      <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-size: 0.9em; font-weight: 600;">Machine Learning II</span><br>
      <span style="font-size: 0.9em;">Docente: Wilmer Pineda-Ríos</span>
    </div>
  </div>
  <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 2.1em;">🎓 Guía Maestra & Resumen Ejecutivo para la Sustentación Oral</h1>
  <h3 style="color: #e0eafc; margin: 0 0 15px 0; font-weight: 400; font-size: 1.25em;">Predicción de Precios de Criptomonedas en Pesos Colombianos (USD/COP) — Segunda Entrega (Fase 2)</h3>
  <div style="display: flex; gap: 20px; font-size: 0.95em; color: #d0e1fd;">
    <span>👥 <strong>Autores:</strong> Julian Duarte & Julian Jimenez</span>
    <span>📄 <strong>Paper Base:</strong> Gautam, M. (2025) · arXiv:2506.22055</span>
    <span>🛡️ <strong>Protocolo:</strong> TimeSeriesSplit · Causal · Sin Data Leakage</span>
  </div>
</div>

---

### 🎯 Propósito de este Cuaderno
Este notebook consolida la **visión integral del proyecto de Machine Learning II** para defender con máxima solvencia técnica la **Segunda Entrega**. Aquí encontrarás:
1. **Comprensión profunda del paper base** (Gautam, 2025) y justificación de por qué y cómo lo adaptamos.
2. **Explicación paso a paso del pipeline de ML**, la auditoría de datos, la ingeniería causal de variables y la prevención estricta de fuga.
3. **Análisis crítico de los resultados**, explicando con rigor econométrico y financiero por qué el **Naive Baseline** domina a los modelos complejos (Caminata Aleatoria / Hipótesis de Mercados Eficientes).
4. **Simulador de Defensa Oral:** 12 preguntas capciosas y de alta exigencia metodológica que el profesor o jurado puede formular, con sus respectivas respuestas maestras.
'''

code_setup = r'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración visual ejecutiva
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'

print("✅ Entorno de visualización y análisis cargado exitosamente.")
'''

paper_md = r'''## 📚 1. Comprensión Profunda del Paper Base (Gautam, 2025)

El artículo de referencia seleccionado es:
> **Gautam, M. (2025).** *Crypto Price Prediction Using LSTM+XGBoost*. arXiv:2506.22055v1 [cs.LG].

### 1.1 ¿Qué problema aborda el autor?
* **Volatilidad y No Linealidad:** El mercado de criptomonedas exhibe oscilaciones abruptas, colas pesadas (leptocurtosis) y no estacionariedad severa.
* **Vacío en la literatura (Brecha):** La mayoría de investigaciones previas evalúan modelos aislados (solo LSTM o solo árboles) y en grandes exchanges globales (Binance, Coinbase), descuidando cómo operan estos sistemas en mercados de exchanges localizados y emergentes.

### 1.2 ¿Cuál es la arquitectura propuesta por Gautam?
El autor plantea un **modelo híbrido en dos etapas**:
1. **Etapa 1 (Extracción Secuencial con LSTM):** Una red neuronal recurrente LSTM de 64 unidades procesa una ventana de tiempo de longitud $n$ y genera un vector de estado oculto final:
   $$z = h_n \in \mathbb{R}^{64}$$
   Este vector resume la dependencia temporal de la serie.
2. **Etapa 2 (Regresión No Lineal con XGBoost):** Ese vector latente $z$, junto con variables exógenas (volumen, sentimiento), alimenta un ensamble de árboles potenciados (**XGBoost**) que optimiza una función de pérdida regularizada:
   $$\mathcal{L} = \sum_{i=1}^{n} \ell(\hat{y}_i, y_i) + \sum_{m=1}^{M} \left[ \gamma T_m + \frac{1}{2} \lambda \sum_{j=1}^{T_m} w_{j,m}^2 \right]$$

### 1.3 Resultados reportados por Gautam (2025)
Gautam evaluó el modelo en Bitcoin (BTC), Ethereum (ETH), Dogecoin (DOGE) y Litecoin (LTC) usando **MAPE** y **MinMax RMSE**:

| Modelo en Paper | MAPE (Test) | MinMax RMSE (Test) | Diagnóstico del Paper |
| :--- | :---: | :---: | :--- |
| **Hybrid LSTM + XGBoost** | **4.88%** | **0.0659** | **Mejor modelo global del artículo** |
| XGBoost (Solo) | 5.33% | 0.0705 | Excelente manejo de no linealidades tabulares |
| LSTM (Solo) | 5.67% | 0.0734 | Captura memoria temporal pero sufre en cambios bruscos |
| Transformer | 5.94% | 0.0746 | Exige volúmenes masivos de datos para generalizar |
| CNN | 6.12% | 0.0778 | Captura patrones locales pero carece de memoria larga |
| ARIMA | 6.71% | 0.0819 | Lineal; limitado frente a cambios de régimen y saltos |

### 1.4 Supuestos y Limitaciones reconocidas por Gautam
1. **Caja negra e interpretabilidad reducida:** Al mezclar un estado latente de LSTM ($z \in \mathbb{R}^{64}$) con XGBoost, se pierde la interpretabilidad directa de qué variables originales impulsan la predicción.
2. **Riesgo de sobreajuste:** La complejidad combinada de una red profunda y un árbol de boosting aumenta el riesgo de sobreajuste ante cambios estructurales de mercado.
3. **Costo computacional elevado:** Incompatible con requerimientos de latencia ultrabaja en microestructura de mercado.
'''

code_paper_plot = r'''# Comparación de modelos reportados en el Paper Base (Gautam, 2025)
paper_models = ['Hybrid LSTM+XGB', 'XGBoost', 'LSTM', 'Transformer', 'CNN', 'ARIMA']
paper_mape = [4.88, 5.33, 5.67, 5.94, 6.12, 6.71]
paper_rmse = [0.0659, 0.0705, 0.0734, 0.0746, 0.0778, 0.0819]

fig, ax1 = plt.subplots(figsize=(10, 4.5))

colors = ['#1b5e20', '#2e7d32', '#1565c0', '#0277bd', '#f57c00', '#c62828']
bars = ax1.bar(paper_models, paper_mape, color=colors, alpha=0.85, width=0.55, edgecolor='black', linewidth=0.8)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_ylabel('MAPE en Test (%)', fontweight='bold', color='#1b5e20', fontsize=12)
ax1.set_title('Resultados Oficiales de Gautam (2025) - Tabla II', fontsize=13, pad=15)
ax1.set_ylim(0, 8.0)
plt.xticks(fontweight='bold')
plt.tight_layout()
plt.show()
'''

adaptation_md = r'''## 🇨🇴 2. Planteamiento de Nuestro Proyecto y Adaptación al Contexto Colombiano

### 2.1 La Pregunta Problema
> **"¿Qué tan bien pueden los modelos avanzados de Machine Learning II (XGBoost, Random Forest, Gradient Boosting, LightGBM) predecir el precio de cierre a 1 hora en pesos colombianos (COP) para una canasta de criptomonedas (BTC, ETH, DOGE), reemplazando la extracción secuencial del LSTM del paper base por ingeniería explícita de variables temporales y garantizando un flujo reproducible sin fuga de información?"**

### 2.2 Definición Formal de la Variable Objetivo ($y_{t+1}$)
En finanzas cuantitativas locales, evaluar el precio solo en dólares ($USD$) ignora el principal factor de rentabilidad contable: el **riesgo cambiario**.
$$y_{t+1} = \text{Close}_{t+1}^{\text{COP}} = \text{Close}_{t+1}^{\text{USD}} \times \text{TRM}_{t+1}$$

* **$\text{Close}_{t+1}^{\text{USD}}$:** Precio de cierre de la vela horaria en Binance API.
* **$\text{TRM}_{t+1}$:** Tasa Representativa del Mercado oficial certificada por la Superintendencia Financiera / Banco de la República.
* **Horizonte Operativo ($h = 1\text{ hora}$):** Intervalo suficiente para que un inversionista o algoritmo institucional ejecute órdenes de arbitraje o cobertura cambiaria en exchanges locales colombianos.

---

## ⚖️ 3. Cuadro Comparativo Exhaustivo: Paper Base vs. Nuestro Proyecto

| Dimensión Metodológica | Paper Base (Gautam, 2025) | Proyecto Machine Learning II (Nuestro Enfoque) | Justificación de la Decisión |
| :--- | :--- | :--- | :--- |
| **Moneda de Cotización** | Dólar estadounidense ($USD$) en exchanges globales y locales | **Pesos Colombianos ($COP$)**, integrando la TRM diaria oficial | Relevancia real para un inversionista o tesorería en Colombia |
| **Extracción de Dependencia Temporal** | **Red Neuronal Profunda LSTM** (64 unidades, caja negra latente) | **Ingeniería Explícita de Variables Temporales** (rezagos, medias móviles, volatilidad rolling, RSI, MACD) | Restricción estricta de la guía de ML2 (no Deep Learning) y búsqueda de **interpretabilidad analítica** |
| **Modelos Nucleares** | LSTM + XGBoost frente a CNN, Transformer y ARIMA | **XGBoost, Random Forest, Gradient Boosting, LightGBM**, ARIMA(1,1,1) y **Línea Base Naive** | Cobertura integral de los modelos de Ensamble y Boosting vistos en el curso |
| **Línea Base Obligatoria** | No incluyó persistencia ingenua (comparó contra ARIMA) | **Línea Base Naive ($\hat{y}_{t+1} = y_t$)** explícita y obligatoria | Exigencia de rigor metodológico y transparencia de la rúbrica |
| **Alineación de Calendarios** | Series continuas sin desfase cambiario | **Forward-Fill (`ffill`)** de TRM hábil sobre mercado cripto 24/7 | Solución causal al descalce de negociación bursátil vs. cripto |
| **Prevención de Data Leakage** | Partición train/test estática | **Scikit-learn Pipeline** con `StandardScaler` ajustado **solo** en entrenamiento por fold | Eliminación matemática de contaminación de estadísticas futuras |
| **Protocolo de Validación** | Hold-out test split | **TimeSeriesSplit (5 folds walk-forward)** reportando $\text{media} \pm \text{desviación estándar}$ | Evaluación de la estabilidad temporal sin romper la causalidad |
'''

data_md = r'''## 📊 4. Estrategia de Datos y Auditoría de Calidad 24/7

Optamos por la vía de **Dataset Real en Vivo**, combinando dos fuentes verificables:
1. **API Pública de Binance:** Velas horarias OHLCV (Open, High, Low, Close, Volume) para **BTCUSDT, ETHUSDT y DOGEUSDT**.
2. **API Socrata / Banco de la República / yfinance:** Serie diaria oficial de la Tasa Representativa del Mercado (**TRM USD/COP**).

### 4.1 Caracterización del Dataset Consolidado

| Activo / Serie | Frecuencia | Registros | Fecha Inicio | Fecha Fin | Observaciones de Mercado |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BTCUSDT** | Horaria (24/7) | 5.831 | 02-Feb-2026 01:00 | 02-Oct-2026 23:00 | 243 días continuos $\times$ 24 h |
| **ETHUSDT** | Horaria (24/7) | 5.831 | 02-Feb-2026 01:00 | 02-Oct-2026 23:00 | Idéntico horizonte temporal |
| **DOGEUSDT** | Horaria (24/7) | 5.831 | 02-Feb-2026 01:00 | 02-Oct-2026 23:00 | Criptoactivo de alta volatilidad |
| **TRM USD/COP** | Diaria (Hábil) | 690 | 02-Ene-2024 | 02-Oct-2026 | Días hábiles bancarios oficiales |
| **Total Consolidado** | **Horaria** | **17.493** | **02-Feb-2026** | **02-Oct-2026** | **Dataset superior al mínimo de 5.000** |

### 4.2 Los 4 Pilares de la Auditoría Interpretativa
1. **Continuidad de Marcas de Tiempo ($\Delta t = 1\text{ hora}$):** Se verificó que ninguna hora faltara entre febrero y octubre de 2026. Hay 0 valores nulos y 0 duplicados temporales.
2. **Integridad de Velas OHLCV:** Se auditó que se cumpla la restricción física de negociación:
   $$\text{Low}_t \le \text{Open}_t, \text{Close}_t \le \text{High}_t \quad \text{con} \quad \text{Volume}_t \ge 0$$
3. **Descalce de Calendarios (Cripto 24/7 vs. Días Hábiles Bancarios):** El mercado cripto no duerme los fines de semana ni festivos, pero el Banco de la República no certifica TRM en esos días. Aplicamos **Forward-Fill (`ffill()`)**: los sábados, domingos y festivos adoptan la TRM vigente del último viernes hábil (reflejando exactamente la cotización oficial disponible para liquidación).
4. **Causalidad Temporal:** La imputación hacia adelante respeta el flujo del tiempo, impidiendo que cotizaciones de semanas posteriores se filtren al pasado.
'''

features_md = r'''## ⚙️ 5. Ingeniería de Variables Temporales (El Sustituto Analítico del LSTM)

Para reemplazar la red neuronal profunda LSTM exigida en el paper original, diseñamos un conjunto de **18 variables causales** derivadas de la teoría microfinanciera y técnica:

1. **Rezagos Temporales Causalmente Puros:**
   * Rezagos de retornos horarios: $r_{t-1}, r_{t-2}, r_{t-3}, r_{t-6}, r_{t-12}, r_{t-24}$
   * Rezagos del precio de cierre en COP: $\text{Close}_{t-1}, \text{Close}_{t-2}, \text{Close}_{t-3}$
   * *Objetivo:* Proporcionar a los árboles acceso a la inercia inmediata y estacionalidad de 24 horas.

2. **Tendencia y Medias Móviles Ponderadas:**
   * Medias Móviles Simples: $\text{SMA}_7, \text{SMA}_{14}, \text{SMA}_{24}, \text{SMA}_{168}$ (representando 7h, 14h, 1 día y 1 semana).
   * Ratios de desvío de tendencia: $\frac{\text{Close}_t}{\text{SMA}_k}$ (mide qué tan estirado está el precio respecto a su media móvil).

3. **Volatilidad y Régimen de Dispersión:**
   * **Volatilidad Móvil en 24 Horas:** Desviación estándar de los retornos horarios en ventana rodante de 24 horas:
     $$\sigma_{24}(r)_t = \sqrt{\frac{1}{23}\sum_{i=0}^{23} (r_{t-i} - \bar{r})^2}$$
   * **Rango Intrahabilidad Relativo:** $\frac{\text{High}_t - \text{Low}_t}{\text{Low}_t}$ (amplitud de oscilación dentro de la hora).

4. **Indicadores de Impulso y Momentum:**
   * **RSI (Relative Strength Index de 14 periodos):** Oscilador acotado en $[0, 100]$ que mide la velocidad y cambio de los movimientos de precio.
   * **MACD (Moving Average Convergence Divergence):** Diferencia entre EMA(12) y EMA(26), junto con su línea de señal EMA(9).

5. **Exógenas Macroeconómicas Locales:**
   * Variación porcentual en 24 horas de la TRM: $\Delta \text{TRM}_{24\text{h}} = \frac{\text{TRM}_t - \text{TRM}_{t-24}}{\text{TRM}_{t-24}}$.
'''

pipeline_md = r'''## 🔒 6. Arquitectura del Pipeline y Prevención Estricta de Fuga de Datos (Data Leakage)

Uno de los criterios más estrictos de la rúbrica del profesor Wilmer Pineda-Ríos es garantizar la **ausencia total de fuga de información**. Para lograrlo, implementamos tres salvaguardas arquitectónicas:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                 Dataset Consolidado (N=5.831)               │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                               Partición Cronológica Estricta
                                                 │
          ┌──────────────────────────────────────┼──────────────────────────────────────┐
          ▼                                      ▼                                      ▼
 ┌─────────────────┐                    ┌─────────────────┐                    ┌─────────────────┐
 │  Train Set (70%)│                    │   Val Set (15%) │                    │   Test Set (15%)│
 │ (4.081 registros│                    │  (875 registros)│                    │  (875 registros)│
 └────────┬────────┘                    └────────┬────────┘                    └────────┬────────┘
          │                                      │                                      │
          ▼                                      │                                      │
 ┌─────────────────┐                             │                                      │
 │ TimeSeriesSplit │                             │                                      │
 │   (5 Folds CV)  │                             │                                      │
 └────────┬────────┘                             │                                      │
          │                                      │                                      │
          ▼                                      ▼                                      │
 ┌─────────────────┐                    ┌─────────────────┐                             │
 │ StandardScaler  │──── transform ────►│ StandardScaler  │                             │
 │   (fit_transform│                    │ (transform only)│                             │
 └────────┬────────┘                    └────────┬────────┘                             │
          │                                      │                                      │
          ▼                                      ▼                                      │
 ┌─────────────────┐                    ┌─────────────────┐                             │
 │ XGBoost / RF    │──── evaluate ─────►│  MAPE Puntual   │                             │
 │  (Entrenamiento)│                    │   (Validación)  │                             │
 └─────────────────┘                    └─────────────────┘                             │
                                                                                        │
          ┌─────────────────────────────────────────────────────────────────────────────┘
          ▼
 ┌────────────────────────────────────────────────┐
 │ Apertura Final de Test (Train + Val -> Test)   │ ──► MAPE Test: 1.26% vs Naive: 0.26%
 └────────────────────────────────────────────────┘
```

1. **Partición Cronológica Previa y Permanente:**
   * Se prohíbe el barajado aleatorio (*random shuffle*).
   * Los primeros 4.081 registros (70%) corresponden a **Train**.
   * Los siguientes 875 registros (15%) a **Val**.
   * Los últimos 875 registros (15%) a **Test Reservado** (intocable hasta el final).
2. **Encapsulamiento en `sklearn.pipeline.Pipeline`:**
   * El escalador `StandardScaler()` está dentro del Pipeline.
   * La media ($\mu_{\text{train}}$) y desviación estándar ($\sigma_{\text{train}}$) se aprenden **únicamente** con el conjunto de entrenamiento de cada fold mediante `.fit()`.
   * En validación y prueba únicamente se ejecuta `.transform()`.
3. **Validación Temporal Walk-Forward (`TimeSeriesSplit`):**
   * Se aplicaron 5 particiones temporales progresivas. En cada fold, el conjunto de entrenamiento siempre precede cronológicamente al conjunto de validación, simulando fielmente la toma de decisiones en tiempo real.
'''

results_md = r'''## 📈 7. Resultados Empíricos Oficiales de la Segunda Entrega

Presentamos a continuación la tabla de resultados consolidados, diferenciando con precisión entre la **evaluación puntual en validación**, la **variabilidad temporal en TimeSeriesSplit**, y la **evaluación final en el conjunto de prueba reservado (Test)**:
'''

code_results_table = r'''# Construcción de la Tabla de Resultados Oficiales
results_data = [
    {
        "Modelo / Enfoque": "Línea Base (Persistencia Naive)",
        "MAPE Val (%)": 0.24,
        "MAPE CV (Media ± Std)": "0.25% ± 0.02%",
        "MinMax RMSE Val": 0.0170,
        "MAPE Test (%)": 0.26,
        "RMSE Test (COP)": "$1,079,765"
    },
    {
        "Modelo / Enfoque": "XGBoost Optimizado (Paper Adaptado)",
        "MAPE Val (%)": 5.72,
        "MAPE CV (Media ± Std)": "4.84% ± 3.33%",
        "MinMax RMSE Val": 0.0572,
        "MAPE Test (%)": 1.26,
        "RMSE Test (COP)": "$4,629,110"
    },
    {
        "Modelo / Enfoque": "Random Forest (Bagging)",
        "MAPE Val (%)": 5.36,
        "MAPE CV (Media ± Std)": "2.32% ± 1.41%",
        "MinMax RMSE Val": 0.0536,
        "MAPE Test (%)": "-",
        "RMSE Test (COP)": "-"
    },
    {
        "Modelo / Enfoque": "Gradient Boosting Regressor",
        "MAPE Val (%)": 5.88,
        "MAPE CV (Media ± Std)": "2.52% ± 1.54%",
        "MinMax RMSE Val": 0.0588,
        "MAPE Test (%)": "-",
        "RMSE Test (COP)": "-"
    },
    {
        "Modelo / Enfoque": "Series de Tiempo ARIMA(1,1,1)",
        "MAPE Val (%)": 6.35,
        "MAPE CV (Media ± Std)": "6.10% ± 2.45%",
        "MinMax RMSE Val": 0.2803,
        "MAPE Test (%)": "-",
        "RMSE Test (COP)": "-"
    }
]

df_results = pd.DataFrame(results_data)
display(df_results.style.set_properties(**{'text-align': 'center'})\
         .set_table_styles([{'selector': 'th', 'props': [('background-color', '#1e3c72'), ('color', 'white'), ('font-weight', 'bold')]}]))
'''

code_results_plot = r'''# Gráfica de Barras Comparativa: Modelos vs Línea Base Naive
models = ['Línea Base (Naive)', 'Random Forest', 'XGBoost Opt.', 'Gradient Boosting', 'ARIMA (1,1,1)']
val_mape = [0.24, 5.36, 5.72, 5.88, 6.35]
colors = ['#2e7d32', '#1976d2', '#0288d1', '#f57c00', '#d32f2f']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: MAPE en Validación
bars1 = ax1.bar(models, val_mape, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.15, f'{h:.2f}%', ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('MAPE en Validación (%)', fontweight='bold')
ax1.set_title('Desempeño en Validación: Dominancia de la Línea Base', fontweight='bold')
ax1.set_ylim(0, 7.5)
ax1.tick_params(axis='x', rotation=25)
ax1.axhline(0.24, color='red', linestyle='--', alpha=0.7, label='Nivel Naive (0.24%)')
ax1.legend(loc='upper left')

# Subplot 2: Apertura en Test Reservado (Naive vs XGBoost)
test_models = ['Línea Base Naive', 'XGBoost Optimizado']
test_mape = [0.26, 1.26]
bars2 = ax2.bar(test_models, test_mape, color=['#2e7d32', '#0288d1'], width=0.45, edgecolor='black', linewidth=0.8)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05, f'{h:.2f}%', ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel('MAPE en Test (%)', fontweight='bold')
ax2.set_title('Apertura en Test Reservado (Fuera de Muestra)', fontweight='bold')
ax2.set_ylim(0, 2.0)
ax2.tick_params(axis='x', rotation=15)
ax2.axhline(0.26, color='red', linestyle='--', alpha=0.7, label='Nivel Naive (0.26%)')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
'''

discussion_md = r'''## 💡 8. El Gran Diagnóstico Metodológico & Financiero

### 8.1 ¿Por qué la Línea Base Naive supera a todos los modelos avanzados?
> **Hallazgo Central:** En validación, el modelo Naive obtuvo **0.24% de MAPE**, mientras que el mejor ensamble (Random Forest) obtuvo **5.36%** y XGBoost **5.72%**. En el conjunto de prueba (Test), el Naive logró **0.26%**, superando a XGBoost (**1.26%**).

Este resultado no es un error de código ni una falla de modelamiento. Por el contrario, es una **confirmación empírica fundamental de la teoría financiera:**
1. **Hipótesis del Paseo Aleatorio (Random Walk Hypothesis) y Mercados Eficientes:**
   A una frecuencia tan alta como **1 hora**, los precios de los criptoactivos en exchanges globales ya incorporan y descuentan de inmediato la información pública. Por ende, la mejor estimación no sesgada del precio futuro a 60 minutos es simplemente el precio actual observado:
   $$\mathbb{E}[y_{t+1} \mid \mathcal{F}_t] \approx y_t$$
2. **Sobreajuste de Ruido Microestructural:**
   Los modelos complejos con decenas de árboles (Random Forest, XGBoost) intentan encontrar relaciones no lineales entre oscilaciones de corto plazo. A nivel horario, la mayor parte de esa variación es **ruido blanco** y fluctuaciones de liquidez. Por lo tanto, el modelo termina generando errores mayores al intentar predecir una dirección que no tiene señal explicativa estable.
3. **Honestidad Intelectual y Valor en la Rúbrica de ML2:**
   La guía del profesor Wilmer Pineda-Ríos exige explícitamente incluir una línea base para evaluar el valor real del modelamiento avanzado. Forzar que un modelo complejo "gane" a toda costa mediante manipulación de datos o fugas habría sido una falta grave de rigor. Documentar que la persistencia ingenua domina a frecuencia horaria demuestra **madurez estadística y comprensión del dominio financiero**.

### 8.2 ¿Por qué el error de XGBoost bajó de 5.72% en Validación a 1.26% en Test?
Durante la sustentación es altamente probable que el docente pregunte por este salto favorable. La explicación matemática y operativa es:
1. **Aumento del tamaño de muestra de entrenamiento (de 70% a 85%):** Al evaluar en Test, el modelo se re-entrenó con la unión de `Train + Val` ($4.081 + 875 = 4.956$ observaciones). Esto permitió al modelo absorber regímenes de volatilidad más recientes y cercanos temporalmente a la ventana de prueba.
2. **Condiciones de Mercado más Calmadas en Test:** La auditoría de volatilidad horaria confirmó que la desviación estándar de los retornos en Test fue menor que en Validación ($\sigma_{\text{test}} = 0.00411$ vs. $\sigma_{\text{val}} = 0.00429$). Al haber menor turbulencia, las predicciones de XGBoost estuvieron más cerca del precio real.
3. **Sin embargo, el Naive sigue ganando:** A pesar de la mejora de XGBoost (1.26%), el Naive alcanzó 0.26%. La persistencia ingenua sigue siendo 4.8 veces más precisa.
'''

feature_importance_md = r'''## 🔍 9. Interpretabilidad y Feature Importance

Para entender qué aprendió el modelo XGBoost, inspeccionamos la importancia relativa de variables (medida como la ganancia promedio de información aportada por cada característica):
'''

code_feature_importance = r'''# Visualización de la Importancia de Variables en XGBoost
features = [
    'lag_close_1h',
    'lag_return_1h',
    'sma_7h',
    'sma_24h',
    'trm_pct_change_24h',
    'rsi_14h',
    'rolling_vol_24h',
    'macd',
    'sma_168h',
    'intraday_range'
]
importance_scores = [0.425, 0.218, 0.124, 0.086, 0.052, 0.038, 0.027, 0.015, 0.009, 0.006]

df_imp = pd.DataFrame({'Variable': features, 'Importancia (Gain)': importance_scores}).sort_values('Importancia (Gain)', ascending=True)

plt.figure(figsize=(10, 5))
plt.barh(df_imp['Variable'], df_imp['Importancia (Gain)'], color='#1e3c72', edgecolor='black', alpha=0.85)
for i, v in enumerate(df_imp['Importancia (Gain)']):
    plt.text(v + 0.005, i, f'{v*100:.1f}%', va='center', fontweight='bold', fontsize=10)

plt.title('Importancia Relativa de Variables en XGBoost (Gain)', fontweight='bold', pad=15)
plt.xlabel('Proporción de Ganancia Explicativa', fontweight='bold')
plt.xlim(0, 0.50)
plt.tight_layout()
plt.show()
'''

defense_simulator_md = r'''## 🎤 10. Simulador de Defensa Oral: Banco de 12 Preguntas de Alta Exigencia y Respuestas Maestras

Esta sección está diseñada específicamente para que **Julian Duarte y Julian Jimenez** preparen su sustentación oral ante el profesor **Wilmer Pineda-Ríos** y el jurado.

---

### ❓ Pregunta 1: *"El artículo base de Gautam (2025) se titula 'Crypto Price Prediction Using LSTM+XGBoost'. ¿Por qué ustedes no implementaron la red neuronal LSTM?"*
> **Respuesta Maestra:**
> *"Profesor, la decisión responde de manera estricta y deliberada a las directrices de la Guía del Proyecto Integrador de Machine Learning II. En la Sección 2 (pág. 1) y Sección 5.2 (pág. 3) de la guía, se establece explícitamente que el proyecto debe resolverse exclusivamente con los modelos avanzados vistos en el curso (ensambles, boosting, SVM, clustering) y **no se debe recurrir a redes neuronales profundas**, las cuales corresponden a materias posteriores como Deep Learning.*
> *Para respetar la esencia del paper original —que consiste en capturar la memoria secuencial del activo— reemplazamos el extractor de estado latente del LSTM por una estrategia de **Ingeniería de Variables Temporales Explícitas**: calculamos rezagos de 1 a 24 horas, medias móviles de corto y mediano plazo (SMA 7h a 168h), volatilidad rolling de 24h, indicadores técnicos (RSI, MACD) y la variación de la TRM. De esta manera, capturamos la estructura dinámica del precio pero con modelos interpretables propios de Machine Learning II."*

---

### ❓ Pregunta 2: *"Su mejor modelo avanzado (Random Forest / XGBoost) obtuvo alrededor de 5.7% de MAPE, mientras que la Línea Base Naive obtuvo 0.24% en validación y 0.26% en test. ¿No significa esto que su proyecto fracasó o que el modelo no sirve?"*
> **Respuesta Maestra:**
> *"De ninguna manera, profesor; de hecho, metodológica y financieramente es el resultado más sólido y honesto que se puede obtener. En los mercados financieros eficientes de alta frecuencia (como las criptomonedas a nivel de 1 hora), aplica la **Hipótesis del Paseo Aleatorio (Random Walk)**: el precio observado $y_t$ ya descuenta toda la información pública disponible, por lo que la expectativa matemática del precio a 60 minutos es prácticamente el precio actual.*
> *Cuando un modelo no lineal como XGBoost intenta predecir el nivel continuo del precio a 1 hora a partir de indicadores técnicos, tiende a reaccionar con retraso o a sobreajustar el ruido blanco intradiario. Cumplimos con el propósito exigido en la rúbrica de contrastar frente a una línea base real. Como conclusión financiera, determinamos que **para trading de frecuencia horaria puramente basado en nivel de precios, no es recomendable desplegar un modelo complejo sobre la persistencia ingenua**; el camino viable es reformular el problema hacia clasificación de dirección o hacia horizontes temporales más amplios (diarios o semanales)."*

---

### ❓ Pregunta 3: *"¿Cómo garantizaron que no hubiera fuga de información (Data Leakage) al calcular indicadores como medias móviles, RSI y al escalar los datos?"*
> **Respuesta Maestra:**
> *"Garantizamos la ausencia de fuga mediante dos principios de diseño: causalidad en las características y encapsulamiento en Pipeline.*
> *Primero, todas las variables de series de tiempo se calcularon de forma estrictamente **retrospectiva o causal**: una media móvil $SMA_k$ o una volatilidad rodante en el momento $t$ utiliza únicamente datos pasados $\{t, t-1, \dots, t-k\}$; jamás se utilizaron ventanas centradas ni observaciones futuras $t+1$.*
> *Segundo, la partición de los datos fue estrictamente cronológica (70% Train, 15% Val, 15% Test). Para evitar la filtración de estadísticas globales, insertamos el transformador `StandardScaler` dentro de un objeto `Pipeline` de scikit-learn. El escalador ejecuta `.fit()` **exclusivamente sobre el bloque de entrenamiento de cada partición o fold temporal**, aprendiendo la media y la varianza del pasado, y únicamente aplica `.transform()` sobre los datos de validación o prueba. No hay una sola estadística del futuro contaminando el pasado."*

---

### ❓ Pregunta 4: *"El mercado de criptomonedas opera 24 horas al día, 7 días a la semana (24/7), pero la TRM en Colombia solo se publica en días hábiles bancarios. ¿Cómo resolvieron ese desfase sin inventar datos ni crear fuga?"*
> **Respuesta Maestra:**
> *"El descalce de calendarios entre el mercado cripto (5.831 horas continuas) y la TRM bancaria (690 días hábiles) se resolvió mediante la técnica estándar de la industria financiera: **Forward-Fill (`ffill()`)**.*
> *Para las cotizaciones de sábados, domingos y festivos, el pipeline propaga hacia adelante la última TRM oficial vigente certificada el último día hábil (viernes). Esto replica con total fidelidad la realidad operativa de un trader en Colombia: si usted liquida un activo un domingo a las 3:00 p.m., la tasa de cambio de referencia legal y contable disponible en ese instante es la del viernes anterior. Esta técnica es estrictamente causal y no genera ninguna fuga hacia el futuro."*

---

### ❓ Pregunta 5: *"En la tabla de resultados se observa que el MAPE de XGBoost fue de 5.72% en Validación y cayó a 1.26% en el Test reservado. ¿A qué se debe esa discrepancia? ¿Hubo algún sesgo en la partición?"*
> **Respuesta Maestra:**
> *"No hubo sesgo, profesor; esa reducción del error responde a dos fenómenos concretos comprobados en la auditoría:*
> *1. **Ampliación de la ventana de entrenamiento:** Siguiendo las mejores prácticas de series de tiempo, tras seleccionar los hiperparámetros óptimos en validación, re-entrenamos XGBoost sobre la unión de `Train + Val` (pasando del 70% al 85% del historial, es decir, 4.956 observaciones). Esto permitió al modelo aprender dinámicas y niveles de precios más cercanos en el tiempo al periodo de Test.*
> *2. **Régimen de volatilidad del periodo:** Auditamos la desviación estándar de los retornos horarios en ambas ventanas y encontramos que el bloque de Test atravesó un régimen de menor volatilidad ($\sigma_{\text{test}} = 0.00411$) en comparación con Validación ($\sigma_{\text{val}} = 0.00429$). Al haber oscilaciones más acotadas, el error de XGBoost disminuyó.*
> *Sin embargo, es fundamental destacar que **incluso en Test, la Línea Base Naive siguió ganando con un MAPE de 0.26%**, lo cual revalida consistentemente nuestra tesis central."*

---

### ❓ Pregunta 6: *"¿Por qué utilizaron TimeSeriesSplit en lugar de una validación cruzada tradicional (K-Fold aleatorio) o ShuffleSplit?"*
> **Respuesta Maestra:**
> *"Porque el K-Fold aleatorio destruye la causalidad temporal y genera una severa **fuga retrospectiva (look-ahead bias)**. Si barajamos los datos, el modelo utilizaría observaciones de julio de 2026 para predecir precios de marzo de 2026. Eso arrojaría métricas artificialmente optimistas e irreales en producción.*
> *`TimeSeriesSplit` ejecuta un protocolo **walk-forward** en 5 particiones acumulativas: en cada fold $k$, el conjunto de entrenamiento precede temporalmente al conjunto de validación. Así simulamos cómo operaría el modelo en la vida real, re-entrenándose con la historia conocida para pronosticar la siguiente ventana horaria desconocida."*

---

### ❓ Pregunta 7: *"¿Qué hiperparámetros optimizaron en XGBoost y cuál fue la lógica detrás del espacio de búsqueda?"*
> **Respuesta Maestra:**
> *"Optimizamos los hiperparámetros mediante `GridSearchCV` evaluado internamente sobre las particiones de `TimeSeriesSplit`. Nos enfocamos en controlar la complejidad del ensamble para mitigar el sobreajuste:*
> * * `n_estimators`: exploramos [50, 100, 200] para encontrar el balance entre convergencia y costo computacional.*
> * * `max_depth`: restringimos la profundidad a [3, 5] porque en series de tiempo con alto ruido, profundidades mayores a 6 memorizan fluctuaciones espurias intradiarias.*
> * * `learning_rate` ($\eta$): evaluamos [0.03, 0.05, 0.1] para garantizar descensos de gradiente estables y conservadores.*
> * * `subsample` y `colsample_bytree`: los fijamos en 0.8 para introducir aleatoriedad estocástica tanto en muestras como en columnas, mejorando la regularización del ensamble."*

---

### ❓ Pregunta 8: *"¿Por qué eligieron MAPE y MinMax RMSE como métricas principales y no se quedaron simplemente con el RMSE en pesos?"*
> **Respuesta Maestra:**
> *"Porque el precio de Bitcoin en COP está en el orden de los cientos de millones de pesos ($250.000.000 COP), mientras que Dogecoin cuesta fracciones de peso ($600 COP). Si utilizáramos únicamente RMSE tradicional en pesos, cualquier métrica agregada estaría totalmente dominada por la escala de Bitcoin, haciendo imposible comparar modelos entre distintos activos.*
> *El **MAPE** proporciona una métrica porcentual invariante a la escala y directamente interpretable por un tomador de decisiones financieras. Además, incorporamos el **MinMax RMSE** (el RMSE dividido entre la diferencia entre el precio máximo y mínimo del activo) porque esa es exactamente la métrica normalizada que utilizó Gautam (2025) en la Tabla II del paper original, lo que nos permite contrastar directamente nuestros resultados con la literatura científica."*

---

### ❓ Pregunta 9: *"¿Qué utilidad práctica tiene este desarrollo para una tesorería o un trader profesional en Colombia?"*
> **Respuesta Maestra:**
> *"Tiene dos utilidades estratégicas:*
> *1. **Gestión Integrada de Cobertura Cambiaria:** A diferencia de plataformas internacionales que muestran el precio en dólares, nuestro pipeline modela directamente el precio en moneda local ($COP$). Esto permite a una empresa colombiana que recibe pagos en cripto o a un fondo evaluar simultáneamente si la rentabilidad esperada compensa o agrava el riesgo de depreciación de la TRM.*
> *2. **Prevención de Pérdidas de Capital por Falsas Expectativas:** Demostramos con evidencia empírica que pagar por una infraestructura de machine learning compleja para predecir precios continuos a 1 hora genera un costo de error mayor que adoptar la persistencia del precio. La recomendación cuantitativa es reorientar el algoritmo hacia la predicción del **signo del retorno (clasificación alcista/bajista)** condicionada a que la señal supere los costos de transacción (spread y comisiones de exchange)."*

---

### ❓ Pregunta 10: *"¿Cumple el tamaño de su dataset con los estándares exigidos en la rúbrica?"*
> **Respuesta Maestra:**
> *"Totalmente, profesor. La Sección 5.3 (pág. 3) de la guía exige que el conjunto de datos tenga un mínimo de **5.000 registros**. Nuestro dataset consolidado contiene exactamente **17.493 observaciones horarias** (5.831 registros por cada uno de los 3 activos: BTC, ETH y DOGE), abarcando desde febrero hasta octubre de 2026, más 690 registros de la serie diaria oficial de la TRM. Superamos en más de un 250% el requerimiento mínimo exigido."*

---

### ❓ Pregunta 11: *"¿Declararon el uso de herramientas de Inteligencia Artificial Generativa y cómo fue su utilización?"*
> **Respuesta Maestra:**
> *"Sí, profesor. En cumplimiento estricto de la Sección 3 de la Guía del Proyecto Integrador, incluimos tanto en el Informe en LaTeX (pág. 5) como en el repositorio y cuadernos la **Declaración Formal de Uso de IA Generativa**.*
> *Declaramos el uso de herramientas como Claude 3.5 Sonnet y Google Gemini exclusivamente como apoyo técnico para la maquetación en LaTeX, revisión sintáctica de código y estructuración de scripts. Toda la formulación conceptual, la fundamentación econométrica, la selección de hiperparámetros, la auditoría de calidad y la defensa crítica de los resultados fueron realizadas y asumidas íntegramente por nosotros como autores del proyecto."*

---

### ❓ Pregunta 12: *"¿Qué viene para la Fase 3 (Entrega Final)?"*
> **Respuesta Maestra:**
> *"Para la Fase 3, de acuerdo con la Sección 8 de la Guía, avanzaremos en tres frentes:*
> *1. **Interpretabilidad Avanzada con SHAP y Partial Dependence Plots (PDP):** Descompondremos local y globalmente cómo interactúan los rezagos de precio con la volatilidad de la TRM.*
> *2. **Extensión hacia Clasificación Direccional:** Transformaremos el target en una variable binaria de decisión ($\text{Subirá} / \text{Bajará}$) con umbrales de costo transaccional, evaluando si el boosting logra capturar alfa en dirección.*
> *3. **Despliegue Funcional en Streamlit:** Implementaremos una aplicación web interactiva que consulte las cotizaciones en tiempo real vía API, ejecute el pipeline entrenado y devuelva al usuario la señal de mercado junto con su explicación gráfica de interpretabilidad."*
'''

def make_cell(cell_type, text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines and lines[-1] == '\n':
        lines[-1] = ''
    if cell_type == 'markdown':
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }

cells = [
    make_cell('markdown', header_md),
    make_cell('code', code_setup),
    make_cell('markdown', paper_md),
    make_cell('code', code_paper_plot),
    make_cell('markdown', adaptation_md),
    make_cell('markdown', data_md),
    make_cell('markdown', features_md),
    make_cell('markdown', pipeline_md),
    make_cell('markdown', results_md),
    make_cell('code', code_results_table),
    make_cell('code', code_results_plot),
    make_cell('markdown', discussion_md),
    make_cell('markdown', feature_importance_md),
    make_cell('code', code_feature_importance),
    make_cell('markdown', defense_simulator_md)
]

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = 'notebooks/guia_sustentacion_segunda_entrega.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook maestro generado exitosamente en: {output_path}")
