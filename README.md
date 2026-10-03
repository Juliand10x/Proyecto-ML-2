# Proyecto Integrador Machine Learning II — Segunda Entrega
## Predicción de Precios de Criptomonedas en Pesos Colombianos (COP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Fase%202%20Completada-brightgreen.svg)]()

**Universidad Externado de Colombia**  
**Facultad de Economía / Maestría en Inteligencia de Negocios & Ciencia de Datos**  
**Asignatura:** Machine Learning II  
**Profesor:** Wilmer Pineda-Ríos  
**Autores:** Julian Duarte & Julian Jimenez  

---

## 📌 Descripción del Proyecto

Este repositorio contiene el código, experimentos e informe formal correspondientes a la **Segunda Entrega (Fase 2: Datos, modelamiento y validación)** del proyecto integrador.

El proyecto adapta el artículo científico de **Gautam (2025)** (*Crypto Price Prediction Using LSTM+XGBoost*), sustituyendo el extractor de memoria profunda (LSTM) por una estrategia dedicada de **Ingeniería de Variables Temporales y Técnicas** (rezagos de 1h a 24h, medias móviles $SMA_7$-$SMA_{168}$, volatilidad rolling de 24h, RSI-14, MACD y la variación porcentual de la TRM dólar/peso).

El objetivo es predecir el precio de cierre a 1 hora en moneda local ($COP$) para una canasta de criptomonedas (**BTCUSDT, ETHUSDT, DOGEUSDT**), utilizando un pipeline sin fuga de información (*data leakage*) y evaluando ensambles avanzados (**XGBoost, Random Forest, Gradient Boosting, LightGBM**) frente a la **Línea Base Naive de persistencia ingenua** ($\hat{y}_{t+1} = y_t$) y modelos econométricos **ARIMA(1,1,1)**.

---

## 📂 Estructura del Repositorio

```bash
Proyecto-ML-2/
├── data/                            # Directorio para almacenamiento de datos (git-ignored parquet/csv)
│   └── .gitkeep
├── docs/                            # Documentación del proyecto y artículos de referencia
│   ├── Crypto price prediction using lstm+xgboost.pdf   # Artículo base (Gautam, 2025)
│   ├── ML2 - Proyecto.pdf           # Rúbrica y guía oficial del proyecto
│   └── primera_entrega.md          # Documentación de la Fase 1
├── src/                             # Código fuente modular en Python
│   ├── data_loader.py               # Extracción desde Binance API & TRM Socrata/yfinance, auditoría 24/7 y ffill
│   ├── feature_engineering.py       # Rezagos, rolling SMA, volatilidad 24h, RSI-14, MACD y exógenas TRM
│   └── pipeline.py                  # Scikit-learn Pipeline con StandardScaler y partición temporal estricta
├── scripts/                         # Scripts auxiliares para la generación del notebook
│   ├── build_complete_nb.py
│   ├── create_nb.py
│   └── run_direct_nb_outputs.py
├── informe_proyecto_ml2.tex         # Informe en LaTeX (3-5 páginas) listo para Overleaf / pdflatex
├── notebook_segunda_entrega.ipynb   # Notebook ejecutable con auditorías, validación walk-forward e interpretación
├── requirements.txt                 # Dependencias del entorno de ejecucion Python
├── .gitignore                       # Filtros de exclusión de git
└── README.md                        # Descripción y guía de ejecución del repositorio
```

---

## ⚙️ Instalación y Requisitos

### 1. Clonar el repositorio
```bash
git clone https://github.com/Juliand10x/Proyecto-ML-2.git
cd Proyecto-ML-2
```

### 2. Crear y activar entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Ejecución del Proyecto

### 1. Ejecutar el Notebook Completo
Puedes abrir y ejecutar el notebook interactivo directamente:
```bash
jupyter notebook notebook_segunda_entrega.ipynb
```

### 2. Ejecución Modular en Python
También puedes ejecutar los módulos de la carpeta `src/` independientemente:

```bash
# Extracción de datos y auditoría
python3 src/data_loader.py

# Construcción de variables temporales
python3 src/feature_engineering.py

# Ejecución del pipeline de entrenamiento y evaluación
python3 src/pipeline.py
```

---

## 📊 Principales Hallazgos Metodológicos

| Modelo / Evaluación | MAPE Val (Puntual) | MAPE CV ($Mean \pm Std$) | MAPE Test (Reservado) | RMSE Test (COP) |
| :--- | :---: | :---: | :---: | :---: |
| **Línea Base Naive ($\hat{y}_{t+1} = y_t$)** | **0.24%** | N/A | **0.26%** | **$1,079,765** |
| Random Forest | 5.36% | $2.32\% \pm 1.41\%$ | - | - |
| Gradient Boosting | 5.88% | $2.52\% \pm 1.54\%$ | - | - |
| XGBoost Optimizado | 5.72% | $4.84\% \pm 3.33\%$ | 1.26% | $4,629,110 |
| ARIMA(1,1,1) | 6.35% | N/A | - | - |

> 💡 **Conclusión Financiera Clave:** Ningún modelo avanzado (XGBoost, Random Forest, ARIMA) logra superar a la **Línea Base Naive de persistencia ingenua** a frecuencia horaria. Este resultado respalda empíricamente la **Hipótesis de Caminata Aleatoria (Random Walk Hypothesis)** en mercados cripto de alta frecuencia y valida la honestidad del protocolo de evaluación contra *baseline* exigido en la rúbrica del curso.

---

## 🤖 Declaración de Uso de IA Generativa
Se declara el uso de herramientas de Inteligencia Artificial Generativa (**Anthropic Claude 3.5 Sonnet** y **Google Gemini 1.5 / 2.0**) como apoyo técnico para la estructuración de código Python, formateo LaTeX e inspección sintáctica. Toda la fundamentación metodológica, decisiones de modelamiento e interpretación financiera fueron realizadas y validadas directamente por el equipo de estudiantes.
