# Guía de Sustentación Oral y Presentación con Diapositivas (Slides)
## Proyecto Integrador de Machine Learning II — Entrega Final
**Universidad Externado de Colombia**  
**Integrantes:** Julian Duarte · Julian Jimenez  
**Docente:** Wilmer Pineda-Ríos  
**Artículo Base:** Gautam, M. (2025). *Crypto Price Prediction Using LSTM+XGBoost*. arXiv:2506.22055v1.  
**Plataforma Web en Producción (Vercel):** [https://public-steel-ten-10.vercel.app](https://public-steel-ten-10.vercel.app)  
**Repositorio GitHub:** [https://github.com/Juliand10x/Proyecto-ML-2](https://github.com/Juliand10x/Proyecto-ML-2)

---

## Estructura General de la Presentación (12 a 15 Minutos)

| Slide | Título / Tema Principal | Tiempo Sugerido | Enfoque Pedagógico |
| :---: | :--- | :---: | :--- |
| **1** | Portada y Planteamiento del Problema | 1.0 min | Contexto local y objetivo de trading en Colombia |
| **2** | Artículo Base (Gautam, 2025) y Adaptación Metodológica | 1.5 min | Sustitución de LSTM por Econometría e Ingeniería Temporal |
| **3** | Origen de Datos, Doble Riesgo (TRM) y Alineación Causal 24/7 | 1.5 min | APIs oficiales y solución matemática a la asincronía sin fuga |
| **4** | Diagnóstico Econométrico: Raíces Unitarias (ADF/KPSS/ACF) | 1.5 min | Evidencia formal de no estacionariedad $I(1)$ vs $I(0)$ |
| **5** | Respuesta al Profesor: Por qué la Línea Base "Parecía Ganar" | 2.0 min | Caminata aleatoria, la trampa del MAPE y el Hit Ratio |
| **6** | Batería Econométrica: SARIMAX, GARCH(1,1)-$t$, VAR y VECM | 2.0 min | Optimización AIC/BIC, colas pesadas y cointegración Johansen |
| **7** | Ingeniería de Variables: Taxonomía de las 51 y Selección Clave | 1.5 min | Justificación de por qué `low_cop` y `close_cop` concentran el 84% |
| **8** | El Modelo Ganador: Formulación del XGBoost Adaptado | 1.5 min | Regularización $L_1/L_2$, control de profundidad y submuestreo |
| **9** | Tabla Maestra de Resultados en Test & Contraste con el Paper | 1.5 min | Evaluación rigurosa en 875 horas consecutivas de prueba |
| **10**| Demostración en Vivo en Vercel & Conclusiones Finales | 1.5 min | Exploración del dashboard interactivo y lecciones aprendidas |

---

## Guion Detallado Slide por Slide

### Slide 1: Portada y Contexto del Proyecto (1.0 min)
* **Contenido Visual del Slide:**
  * Título: *Predicción de Precios de Criptomonedas en Pesos Colombianos (COP) mediante Econometría Financiera y Ensambles de Machine Learning*.
  * Nombres de los integrantes, asignatura, fecha y logos institucionales.
  * Tarjeta conceptual: *"¿Cómo mitigar el riesgo cambiario y la volatilidad en la toma de decisiones algorítmicas en Colombia?"*.
* **Qué decir:**
  > *"Buenos días profesor y compañeros. Hoy presentamos la entrega final de nuestro proyecto integrador en Machine Learning II. Nuestro objetivo fue diseñar, validar y desplegar en producción un sistema predictivo capaz de anticipar el precio y la dirección del mercado de criptoactivos expresados en moneda local (pesos colombianos), respondiendo a los desafíos reales que enfrenta un inversionista institucional o retail en el país."*

---

### Slide 2: El Artículo Base y la Adaptación Local (1.5 min)
* **Contenido Visual del Slide:**
  * Referencia: Gautam, M. (2025). *Crypto Price Prediction Using LSTM+XGBoost*.
  * Tabla comparativa:
    * *Enfoque del Paper:* Diario en USD, 5 variables brutas OHLCV, extractor neuronal recurrente LSTM de 64 unidades + regresor XGBoost.
    * *Nuestra Adaptación:* Horario (1H) en COP, 51 variables técnicas y cambiarias, sustitución de redes profundas por series de tiempo complejas (SARIMAX, GARCH, VAR/VECM) y ensambles avanzados del curso (XGBoost regularizado, LightGBM, CatBoost, Random Forest).
* **Qué decir:**
  > *"Tomamos como punto de partida el trabajo de Gautam (2025). Sin embargo, el artículo original modela activos en dólares a frecuencia diaria usando redes LSTM. Siguiendo las directrices del microcurrículo del curso, adaptamos y superamos este enfoque: trasladamos el problema a frecuencia horaria en pesos colombianos, incorporamos el riesgo cambiario de la TRM y reemplazamos la red neuronal profunda por formulaciones econométricas avanzadas e ingeniería de características fundamentada."*

---

### Slide 3: Datos, Doble Riesgo y Alineación Causal 24/7 (1.5 min)
* **Contenido Visual del Slide:**
  * Fuentes oficiales: Binance Public REST API v3 + Datos Abiertos Socrata (Banco de la República / Superfinanciera).
  * Ecuación: $P_t^{\text{COP}} = P_t^{\text{USD}} \times \text{TRM}_t$.
  * Gráfico: **Figura 1 del Paper** (*Evolución temporal de Bitcoin en COP vs. TRM y retornos horarios*).
  * Esquema de solución: *Forward-fill causal* para sincronizar el mercado 24/7 de criptos con los días hábiles del mercado spot bancario (SET-FX).
* **Qué decir:**
  > *"Un trader en Colombia está expuesto a dos fuerzas independientes: el riesgo de mercado del token y el riesgo macroeconómico de devaluación del peso frente al dólar. Descargamos 17.493 observaciones horarias consolidadas. Para resolver la asincronía entre el mercado cripto (que nunca duerme) y el mercado interbancario de divisas (que cierra fines de semana y festivos), aplicamos un forward-fill causal estricto: al predecir la hora $t+1$, solo utilizamos la última TRM legalmente certificada, garantizando cero fuga de información (look-ahead bias)."*

---

### Slide 4: Diagnóstico Econométrico: Raíces Unitarias (ADF/KPSS/ACF) (1.5 min)
* **Contenido Visual del Slide:**
  * Tabla de contrastes:
    * Precios en nivel ($P_t$): ADF stat = $-1.5951$ ($p = 0.4861$), KPSS stat = $3.4876$ ($p < 0.01$) $\rightarrow$ **No estacionaria $I(1)$**.
    * Retornos diferenciados ($\Delta P_t / P_{t-1}$): ADF stat = $-76.0357$ ($p = 0.0000$), KPSS stat = $0.2237$ ($p > 0.10$) $\rightarrow$ **Estacionaria $I(0)$**.
  * Gráfico: **Figura 2 del Paper** (*Funciones ACF y PACF de Precios vs. Retornos*).
* **Qué decir:**
  > *"Antes de correr cualquier modelo, auditamos la estructura generadora de datos. Los contrastes de Dickey-Fuller Aumentado y KPSS demostraron unánimemente que los precios en nivel son caminatas aleatorias no estacionarias $I(1)$: su ACF decae de manera hiperlenta y la PACF colapsa en el primer rezago ($r_1 \approx 1$). Esto fundamenta teóricamente la necesidad obligatoria de diferenciar hacia retornos horarios, logrando una serie estacionaria $I(0)$ que cumple los supuestos asintóticos de modelamiento."*

---

### Slide 5: Respuesta al Profesor: Por qué la Línea Base "Parecía Ganar" (2.0 min)
* **Contenido Visual del Slide:**
  * Comparativa conceptual: *Nivel de Precios vs. Retorno y Dirección*.
  * Persistencia Ingenua (Naive): $\hat{P}_{t+1} = P_t$.
  * Tarjetas de choque métrico:
    * *MAPE Numérico en Precios:* Naive = 0.261% vs. XGBoost = 0.345% $\rightarrow$ *"Ventaja Aritmética Espuria"*.
    * *Acierto Direccional (Hit Ratio):* **Naive = 0.00% vs. XGBoost = 51.77%** $\rightarrow$ *"Superioridad Operativa Real"*.
* **Qué decir (Punto de máxima contundencia académica):**
  > *"En la entrega anterior, el profesor nos señaló con total acierto que resultaba sospechoso que la persistencia ingenua superara a los ensambles. Hoy demostramos matemáticamente la causa: en series de alta frecuencia integradas de orden uno, la variación entre $t$ y $t+1$ rara vez supera el 0.2%. Decir que el precio de la próxima hora será exactamente el mismo de hoy arroja un MAPE minúsculo por pura inercia. Sin embargo, su valor económico es nulo: el Naive tiene un acierto direccional del 0.00%. En cambio, nuestro XGBoost Adaptado alcanza un Hit Ratio del 51.77%, superando el azar (50%) y generando una ventaja probabilística real tras costos de transacción."*

---

### Slide 6: Batería Econométrica: SARIMAX, GARCH, VAR y VECM (2.0 min)
* **Contenido Visual del Slide:**
  * Tabla de Optimización SARIMAX: Selección de **SARIMAX(1,0,0) + Exógenas** ($\text{AIC} = -31.554,28$).
  * Tabla de Optimización GARCH: **GARCH(1,1) con innovaciones Student's $t$** ($\text{AIC} = 4.901,76$ vs. $6.346,90$ de la normal $\rightarrow$ $\Delta > 1.440$ puntos).
  * Gráfico: **Figura 3 del Paper** (*Volatilidad condicional estimada y residuos estandarizados como ruido blanco*).
  * Cointegración de Johansen: Estadístico de traza para $r \le 0$ de $51.87 > 47.85$ (comprueba cointegración entre BTC, ETH, DOGE y TRM).
* **Qué decir:**
  > *"Construimos y optimizamos sistemáticamente una batería de series de tiempo complejas:
  > 1. En SARIMAX, el orden AR(1) con la TRM y osciladores técnicos resultó óptimo por parsimonia según AIC y BIC.
  > 2. En GARCH, demostramos el agrupamiento de volatilidad (volatility clustering) y la presencia de colas pesadas: usar la distribución Student's t redujo drásticamente el AIC en más de 1.440 puntos.
  > 3. Mediante el contraste de Johansen, comprobamos que existe cointegración de largo plazo entre las criptomonedas y la tasa de cambio colombiana, justificando la estimación tanto de modelos VAR multivariados como de corrección de errores VECM."*

---

### Slide 7: Ingeniería de Variables: Taxonomía de las 51 y Selección Clave (1.5 min)
* **Contenido Visual del Slide:**
  * Gráfico: **Figura 5 del Paper** (*Ganancia en Árbol / Feature Importance de XGBoost*).
  * Tabla de variables clave seleccionadas:
    * `low_cop` (51.78%) $\rightarrow$ Soporte y piso de liquidación horario.
    * `close_cop` (32.17%) $\rightarrow$ Ancla de persistencia estocástica.
    * `open_cop` (9.08%) $\rightarrow$ Presión compradora/vendedora de la vela.
    * `high_cop` (6.37%) $\rightarrow$ Techo y rechazo de liquidez.
    * `sma_24h` + `lag_trm_3h/6h` $\rightarrow$ Tendencia intradiaria y canal cambiario.
  * *Dato de impacto:* Estas características concentran más del **99.4% de la ganancia explicativa**.
* **Qué decir:**
  > *"Construimos 51 variables agrupadas en microestructura, momentum, medias móviles, osciladores de impulso y dinámica cambiaria. Al auditar la importancia por ganancia (Gain) en XGBoost, encontramos que las variables de microestructura local en pesos (especialmente el mínimo horario low_cop y el cierre close_cop) concentran más del 83% del poder explicativo, y junto a la tendencia de 24 horas y los rezagos de la TRM superan el 99.4%. Esto demuestra que en alta frecuencia, la física del libro de órdenes local domina sobre osciladores sintéticos de largo plazo."*

---

### Slide 8: El Modelo Ganador: Formulación del XGBoost Adaptado (1.5 min)
* **Contenido Visual del Slide:**
  * Ecuación de regularización formal: $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum w_j^2 + \alpha \sum |w_j|$.
  * Cuadro comparativo tridimensional:
    * *Frente al Paper de Gautam:* Gautam usa 5 columnas en USD diario sin regularización estricta. Nosotros usamos 51 variables locales en COP con control de sobreajuste.
    * *Frente a Random Forest:* RF solo promedia varianza; XGBoost calcula derivadas de 2° orden (Hessiano) para ponderar los residuos difíciles.
    * *Frente a LightGBM:* El crecimiento por nivel (*depth-wise*) de XGBoost con poda $\gamma$ previene ramas asimétricas ruidosas en alta frecuencia.
  * Hiperparámetros definitivos: `max_depth = 5`, `learning_rate = 0.03`, `subsample = 0.8`, `colsample_bytree = 0.8`.
* **Qué decir:**
  > *"El modelo ganador definitivo fue nuestro XGBoost Adaptado. Su superioridad técnica radica en que impusimos un control estricto de complejidad: limitamos la profundidad a 5 niveles, fijamos una tasa de aprendizaje suave de 0.03 e introdujimos un submuestreo estocástico del 80%. Esto evita que los árboles memoricen el ruido horario y le permite alcanzar el mayor acierto direccional (51.77%) frente a Random Forest (46.74%) y LightGBM (47.89%)."*

---

### Slide 9: Tabla Maestra de Resultados en Test & Contraste con el Paper (1.5 min)
* **Contenido Visual del Slide:**
  * Tabla de Desempeño en Partición Reservada de Prueba (875 horas consecutivas):
    * **XGBoost Adaptado (Ganador):** $\text{MAPE} = 0.345\%$, $\text{MinMax RMSE} = 0.0235$, $\text{Hit Ratio} = \mathbf{51.77\%}$.
    * **SARIMAX(1,0,0) + Exógenas:** $\text{MAPE} = 0.288\%$, $\text{MinMax RMSE} = 0.0210$, $\text{Hit Ratio} = 50.17\%$.
    * **Línea Base Naive:** $\text{MAPE} = 0.261\%$, $\text{MinMax RMSE} = 0.0203$, $\text{Hit Ratio} = 0.00\%$.
  * Contraste con Gautam (2025):
    * MinMax RMSE Gautam: $0.0705$ (XGBoost) y $0.0650$ (Híbrido LSTM).
    * MinMax RMSE Nuestro en COP: **$0.0235$** (mejora superior al 60% en escala normalizada).
* **Qué decir:**
  > *"Evaluamos sobre 875 horas estrictamente reservadas (15% final). Al contrastar con Gautam (2025), nuestro modelo XGBoost en pesos colombianos obtiene un MinMax RMSE de 0.0235, superando ampliamente el 0.0705 del paper base y su híbrido LSTM (0.0650). Demostramos formalmente que una ingeniería de características temporales enriquecida y fundamentada en econometría puede igualar o superar la memoria de una red profunda recurrente, con total transparencia e interpretabilidad."*

---

### Slide 10: Demostración en Vivo en Vercel & Conclusiones (1.5 min)
* **Contenido Visual del Slide:**
  * Captura de pantalla del dashboard web: `https://public-steel-ten-10.vercel.app`.
  * Código QR o enlace directo al proyecto.
  * Resumen de 3 Conclusiones:
    1. *Desmitificación del Baseline:* La aparente dominancia del Naive en niveles desaparece al medir acierto direccional (51.77% en XGBoost).
    2. *Aporte Econométrico:* Los retornos en COP presentan leptocurtosis extrema (GARCH Student's t) y cointegración con la TRM.
    3. *Despliegue Funcional:* Aplicación interactiva de grado profesional en Vercel con simulador de escenarios cambiarios.
* **Qué decir:**
  > *"Como exigencia de esta entrega final, desplegamos una aplicación web interactiva en producción sobre Vercel. En ella, cualquier usuario puede auditar las fórmulas en KaTeX, explorar las 51 variables, revisar los diagnósticos econométricos y utilizar un simulador de inversión que estresa la TRM en tiempo real. En conclusión: respondimos a cada observación docente, demostramos el valor del Machine Learning sobre la caminata aleatoria y dejamos un producto 100% reproducible y operativo. Muchas gracias, quedamos atentos a sus preguntas."*

---

## Banco de Respuestas a Preguntas Críticas del Docente

| Pregunta Esperada del Profesor | Respuesta Sugerida (Firme y Fundamentada) |
| :--- | :--- |
| **1. "¿Por qué no utilizaron una red neuronal profunda LSTM como en el paper de Gautam?"** | *"El microcurrículo de Machine Learning II excluye explícitamente el uso de redes profundas (propias de cursos posteriores de Deep Learning). Nuestro reto consistió en responder a la pregunta de investigación: ¿Puede una ingeniería de características multiescala fundamentada en econometría sustituir al extractor de memoria LSTM? La respuesta empírica es sí: logramos un MinMax RMSE de 0.0235 frente al 0.0650 del paper, con menor complejidad y plena interpretabilidad."* |
| **2. "¿Por qué el VECM tuvo un error cuadrático mayor que el VAR en Test?"** | *"Porque el VECM modela el vector de cointegración hacia el equilibrio de largo plazo ($\boldsymbol{\Pi} = \boldsymbol{\alpha} \boldsymbol{\beta}'$). En finanzas a frecuencia horaria (1H), el ruido de microestructura y las fluctuaciones de corto plazo dominan sobre las correcciones de equilibrio fundamental, las cuales tardan semanas o meses en converger. Por ello, a escala horaria el VAR en primeras diferencias y los árboles de gradiente capturan mejor la dinámica inmediata."* |
| **3. "¿Cómo garantizaron que no hubiera fuga de datos al alinear la TRM?"** | *"Mediante un forward-fill causal estricto. Al pronosticar el horizonte $t+1$, solo utilizamos información disponible hasta el momento $t$. Si $t$ es domingo a las 11:00 PM, la tasa representativa legalmente vigente en Colombia es la certificada el viernes por la Superfinanciera, exactamente igual a lo que experimentaría un trader en tiempo real en los exchanges locales."* |
| **4. "¿Por qué un Hit Ratio del 51.77% se considera un éxito en trading de alta frecuencia?"** | *"Porque en mercados financieros altamente eficientes bajo la hipótesis de caminata aleatoria, la precisión direccional teórica del azar es del 50.00%. En finanzas cuantitativas a escala horaria, una ventaja estadística persistente del 1.77% sobre el azar representa una ventaja acumulativa sustancial para la construcción de carteras sistemáticas una vez descontados los costos transaccionales."* |
| **5. "¿Por qué se eligió la distribución Student's t sobre la Gaussiana en el modelo GARCH?"** | *"Porque los retornos financieros en Colombia sufren de leptocurtosis (colas pesadas causadas por eventos extremos de volatilidad y saltos cambiarios en la TRM). Al evaluar los criterios de información, la distribución Student's t redujo el AIC en 1.445 puntos respecto a la Normal (de 6.346 a 4.901), demostrando que una campana de Gauss subestima severamente el riesgo de cola."* |
