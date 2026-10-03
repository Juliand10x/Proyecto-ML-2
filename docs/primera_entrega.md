# Primera Entrega — Proyecto Integrador
## Asignatura: Machine Learning II · Universidad Externado de Colombia

**Integrantes:** [Nombre 1 — Código] · [Nombre 2 — Código] · [Nombre 3 — Código, si aplica]
**Artículo base:** Gautam, M. (2025). *Crypto Price Prediction Using LSTM+XGBoost*. arXiv:2506.22055v1 [cs.LG].
**Fase:** 1 — Comprensión del artículo y planteamiento

---

## 1. Comprensión y sustentación del artículo

### 1.1 Problema que aborda

El artículo enfrenta el problema de **predecir el precio de criptomonedas** (Bitcoin, Ethereum, Dogecoin y Litecoin) en un contexto de alta volatilidad, dinámicas no lineales y sensibilidad a factores exógenos (sentimiento de mercado, volumen, indicadores macroeconómicos). El autor señala una brecha específica: la mayoría de estudios previos tratan LSTM y XGBoost de forma aislada, y muy pocos trabajos exploran datos de exchanges **localizados** (fuera de los grandes exchanges globales como Binance o Coinbase), a pesar del creciente peso de mercados emergentes en el volumen de operación cripto.

### 1.2 Método propuesto

El modelo es una arquitectura híbrida de **dos etapas**:

1. **Extracción temporal con LSTM:** una red LSTM procesa una ventana de `n` pasos de tiempo con `d` variables (precio y features asociadas) y produce un vector de estado oculto final $z = h_n \in \mathbb{R}^k$ (con $k=64$), que resume la dependencia temporal de la serie.
2. **Regresión no lineal con XGBoost:** ese vector $z$ se usa como entrada de un modelo XGBoost, que aprende $\hat{y} = f(z) = \sum_{m=1}^{M} f_m(z)$, minimizando una función de pérdida regularizada:

$$L = \sum_{i=1}^n \ell(\hat{y}_i, y_i) + \sum_{m=1}^M \Omega(f_m), \qquad \Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^T w_j^2$$

El razonamiento del autor: LSTM captura la memoria temporal de la serie, mientras XGBoost aprovecha esa representación para modelar interacciones no lineales con variables auxiliares (sentimiento, indicadores técnicos, macroeconómicos) que un modelo puramente secuencial no explota tan bien.

### 1.3 Supuestos principales

- Las dependencias temporales relevantes caben en una ventana finita de `n_steps_in` pasos.
- Las variables auxiliares (sentimiento, indicadores técnicos) aportan información no capturada por el precio histórico por sí solo.
- Los datos de exchanges (globales y locales) son representativos del comportamiento real del mercado y están libres de manipulación severa.
- La comparación entre criptomonedas es válida tras normalizar (p. ej., "primer día = 100") porque sus escalas absolutas difieren mucho.

### 1.4 Resultados reportados

Usando MAPE y RMSE normalizado (Min-Max) como métricas, el modelo híbrido obtiene el mejor desempeño frente a LSTM solo, CNN, Transformer, ARIMA y XGBoost solo:

| Modelo | MAPE (test) | MinMax RMSE (test) |
|---|---|---|
| ARIMA | 0.0671 | 0.0819 |
| CNN | 0.0612 | 0.0778 |
| Transformer | 0.0594 | 0.0746 |
| LSTM | 0.0567 | 0.0734 |
| XGBoost | 0.0533 | 0.0705 |
| **Hybrid LSTM+XGBoost** | **0.0488** | **0.0659** |

El análisis exploratorio adicional muestra: distribuciones de retornos diarios leptocúrticas (colas pesadas), volatilidad no estacionaria (ventanas rolling de 30 días), correlaciones crecientes entre criptomonedas en períodos de estrés, y dominancia persistente de Bitcoin en capitalización de mercado.

### 1.5 Limitaciones reconocidas por el autor

1. Riesgo de sobreajuste a condiciones de mercado específicas, dada la complejidad conjunta de LSTM + XGBoost.
2. Interpretabilidad limitada del sistema híbrido de punta a punta, pese a que XGBoost individualmente es más interpretable que una red profunda.
3. Dependencia de señales estáticas (precio e indicadores técnicos), sin incorporar de forma robusta señales no cuantitativas (sentimiento social, noticias, eventos geopolíticos).
4. Costo computacional alto, poco compatible con *trading* de alta frecuencia en tiempo real.

---

## 2. Planteamiento del problema

### 2.1 Pregunta a responder

*¿Es posible predecir con precisión aceptable el retorno/precio de corto plazo de una canasta de criptomonedas relevantes para un inversionista colombiano, replicando la lógica de extracción temporal + regresión no lineal del artículo, pero usando exclusivamente modelos avanzados de Machine Learning II (sin redes neuronales profundas)?*

### 2.2 Tipo de tarea

**Regresión.** Variable objetivo: precio de cierre (o alternativamente, el retorno porcentual) de la criptomoneda a un horizonte corto (p. ej., 1 a 24 horas adelante), replicando el enfoque de forecasting continuo del artículo.

### 2.3 Variable objetivo

`close_price_t+h` (precio de cierre en el horizonte de predicción `h`), o su variante `return_t+h = (close_t+h − close_t) / close_t`, evaluada con las mismas métricas del artículo (MAPE y MinMax RMSE) para permitir comparación directa con los resultados reportados.

### 2.4 Relevancia en el contexto colombiano/latinoamericano

Colombia ha tenido una adopción creciente de criptoactivos como cobertura frente a la volatilidad cambiaria y como alternativa de inversión fuera del sistema financiero tradicional. Sin embargo, la literatura de forecasting cripto rara vez incorpora explícitamente la dinámica **peso colombiano–dólar**, que es la referencia real de decisión para un inversionista local (un retorno positivo en USD puede no serlo en COP si el peso se aprecia, y viceversa). Este proyecto adapta el artículo para responder no solo "¿cuánto valdrá el activo en dólares?", sino "¿cuánto valdrá en términos que le importan a un inversionista en Colombia?", incorporando la tasa de cambio como variable macroeconómica local — exactamente el tipo de predictor auxiliar que el artículo original menciona como relevante pero no explota a fondo.

### 2.5 Qué se reproduce y qué se adapta

| Componente del artículo | Reproducción / Adaptación |
|---|---|
| Extracción temporal (LSTM) | **Se sustituye** por ingeniería de variables temporales explícitas (lags, medias móviles, volatilidad rolling, momentum) — no se usa Deep Learning, según restricción de la guía. Esta sustitución busca capturar el mismo tipo de dependencia temporal que el LSTM aprendería implícitamente. |
| Regresión no lineal (XGBoost) | **Se reproduce directamente.** XGBoost (o Gradient Boosting de scikit-learn) es un modelo avanzado del curso (Boosting) y se usa tal como en el artículo, sobre las variables temporales ya construidas. |
| Variables auxiliares (sentimiento, macro) | **Se adapta:** en vez de sentimiento de redes sociales (fuera de alcance por complejidad de recolección), se usa la tasa de cambio USD/COP como variable macroeconómica local, más volumen y variables técnicas. |
| Métricas de evaluación (MAPE, MinMax RMSE) | **Se reproducen tal cual**, para que los resultados sean comparables con la Tabla II del artículo. |
| Comparación de modelos | **Se amplía:** además del modelo híbrido adaptado, se comparan Random Forest y SVR (kernel) como alternativas del curso, y una línea base de persistencia ingenua. |

---

## 3. Estrategia de datos

### 3.1 Vía elegida: dataset real

Se opta por **datos reales**, combinando dos fuentes públicas y verificables:

1. **Precios históricos de criptomonedas:** API pública de Binance (`klines`/`candlestick` endpoint), sin necesidad de autenticación para datos históricos. Se propone frecuencia horaria para BTC/USDT, ETH/USDT y al menos una tercera criptomoneda de interés (p. ej., BNB/USDT o SOL/USDT), lo que permite alcanzar fácilmente miles de registros por activo en un rango de 1-2 años.
2. **Tasa de cambio USD/COP:** API del Banco de la República de Colombia (series estadísticas, tasa representativa del mercado - TRM), como variable macroeconómica local.

### 3.2 Caracterización del conjunto previsto

| Aspecto | Detalle |
|---|---|
| Variable objetivo | Precio de cierre (o retorno) a horizonte `h` horas, por criptomoneda |
| Predictores previstos | Precio (open, high, low, close), volumen, lags de precio y retorno, medias móviles (7, 14, 30 periodos), volatilidad rolling, TRM USD/COP y su variación, indicadores técnicos simples (RSI, MACD) |
| Unidad de observación | Vela horaria por criptomoneda |
| Tamaño estimado | ≥ 5.000 registros (una sola cripto a frecuencia horaria durante ~8 meses ya supera ese umbral; se ampliará usando 2-3 criptomonedas en panel para robustez) |
| Calidad inicial esperada | Alta: exchanges reportan datos limpios de OHLCV; posible necesidad de alinear zonas horarias y frecuencia entre la fuente cripto (continua, 24/7) y la TRM (solo días hábiles, se propone *forward-fill*) |
| Fuente | Binance API (histórico público) + Banco de la República (series TRM) |

### 3.3 Por qué no se requiere simulación

Ambas fuentes son de acceso público, gratuito y verificable, cubren el período y la frecuencia necesarios, y permiten trazabilidad completa (cualquier persona puede reconstruir el dataset desde las mismas APIs). Por tanto, no se contempla simulación fundamentada para esta entrega; si durante la Segunda entrega se detectan vacíos (p. ej., datos de sentimiento local no disponibles), se evaluará una simulación parcial y justificada en ese momento.

---

## 4. Plan de modelamiento y línea base

### 4.1 Modelos avanzados candidatos

| Modelo | Rol en el proyecto |
|---|---|
| **XGBoost / Gradient Boosting Regressor** | Reproduce directamente la segunda etapa del artículo; modelo principal a evaluar |
| **Random Forest Regressor** | Alternativa de ensamble (Bagging) del curso, para contrastar con Boosting |
| **SVR (Support Vector Regression, kernel RBF)** | Modelo de Máquinas de Soporte Vectorial del curso, para evaluar si una frontera no lineal distinta aporta ventaja |

Las variables temporales generadas manualmente (lags, rolling stats) alimentan a los tres modelos por igual, sustituyendo el rol del LSTM del artículo.

### 4.2 Métrica principal

Se conservan las métricas del artículo para permitir comparación directa:

- **MAPE** (Mean Absolute Percentage Error): interpretable como error porcentual promedio, relevante para comunicar resultados a un inversionista no técnico.
- **MinMax RMSE**: normaliza el error por el rango de precios, permitiendo comparar el desempeño entre criptomonedas de escalas muy distintas (BTC vs. una altcoin de menor valor).

Se prioriza MAPE como métrica principal de decisión por su interpretabilidad directa; MinMax RMSE se reporta como métrica secundaria de robustez frente a outliers.

### 4.3 Estrategia de validación prevista

- Al tratarse de series de tiempo, se usará **validación temporal** (no aleatoria): partición cronológica en entrenamiento/validación/prueba, evitando que información futura contamine el entrenamiento (sin *shuffle*).
- Se prevé validación cruzada con esquema de ventana expansiva o *walk-forward* para el ajuste de hiperparámetros en la Segunda entrega, en lugar de K-Fold estándar, precisamente para prevenir fuga de información temporal.

### 4.4 Línea base

Se construye una **línea base de persistencia ingenua** (naive forecast): $\hat{y}_{t+h} = y_t$, es decir, predecir que el precio en `h` horas será igual al precio actual. Esta es la línea base estándar en forecasting financiero y es consistente con el nivel de Machine Learning I (equivalente conceptual a un "predictor de la última observación", análogo al baseline de mediana usado en talleres previos del curso). Cualquier modelo avanzado debe superar claramente esta referencia para justificar su complejidad adicional.

---

## Nota metodológica sobre el uso de IA generativa

Se declara el uso de Claude (Anthropic) como herramienta de apoyo para la estructuración de este documento y la interpretación inicial del artículo base. Todas las decisiones metodológicas (adaptación del componente LSTM, selección de fuentes de datos, elección de modelos y métricas) fueron discutidas y son sustentables por el equipo.

---

## Pendientes antes de la sustentación

- [ ] Completar nombres y códigos de los integrantes.
- [ ] Verificar acceso y formato exacto de la API de Binance y de la TRM del Banco de la República (probar extracción antes de la Segunda entrega).
- [ ] Confirmar el horizonte de predicción `h` definitivo (1h, 4h, 24h) según la disponibilidad real de datos.
- [ ] Revisar ortografía y gramática de todo el documento antes de entregar (la guía penaliza explícitamente errores).
- [ ] Verificar extensión final: entre 3 y 5 páginas de cuerpo (sin contar portada, bibliografía, gráficos y tablas).
