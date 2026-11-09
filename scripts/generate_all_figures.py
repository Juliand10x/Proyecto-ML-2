"""
Genera todas las figuras estáticas de alta resolución (DPI 300) requeridas para:
1. El informe en LaTeX (informe_proyecto_ml2.tex)
2. El Jupyter Notebook enriquecido visualmente
3. La aplicación web desplegada en Vercel
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

print("[1/5] Cargando datos...")
df = pd.read_csv('data/crypto_trm_featured.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

btc = df[df['symbol'] == 'BTCUSDT'].sort_values('timestamp').reset_index(drop=True)
piv = df.pivot(index='timestamp', columns='symbol', values='close_cop').dropna()
piv['TRM'] = df[df['symbol'] == 'BTCUSDT'].set_index('timestamp')['trm']
piv = piv.dropna()

# -------------------------------------------------------------
# FIGURA 1: Precios en COP y Volatilidad Comparativa
# -------------------------------------------------------------
print("[2/5] Generando Fig 1: Evolución de Precios en COP...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax1.plot(btc['timestamp'], btc['close_cop'] / 1e6, color='#10b981', lw=1.5, label='Bitcoin (BTC/COP en Millones)')
ax1.set_ylabel('Precio (Millones COP)', fontsize=11, fontweight='bold')
ax1.set_title('Evolución Histórica Horaria de Bitcoin y TRM en Colombia (2026)', fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper left', frameon=True)

ax1_trm = ax1.twinx()
ax1_trm.plot(btc['timestamp'], btc['trm'], color='#f59e0b', lw=1.2, ls='--', label='TRM Oficial (USD/COP)')
ax1_trm.set_ylabel('TRM (COP/USD)', color='#f59e0b', fontsize=11, fontweight='bold')
ax1_trm.legend(loc='upper right', frameon=True)
ax1_trm.grid(False)

# Retornos diferenciados
ax2.plot(btc['timestamp'], btc['return_cop_1h'] * 100, color='#0284c7', lw=0.8, alpha=0.85, label='Retorno Horario BTC (%)')
ax2.axhline(0, color='black', lw=0.8, ls=':')
ax2.set_ylabel('Retorno (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Fecha / Hora', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
fig.savefig('figures/fig1_price_trm_returns.png', dpi=300)
fig.savefig('public/figures/fig1_price_trm_returns.png', dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# FIGURA 2: ACF y PACF de Precios (I(1)) vs. Retornos (I(0))
# -------------------------------------------------------------
print("[3/5] Generando Fig 2: ACF y PACF Precios vs Retornos...")
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

plot_acf(btc['close_cop'], lags=30, ax=axes[0, 0], color='#ef4444', title='ACF: Precios en Nivel (Persistencia I(1))')
plot_pacf(btc['close_cop'], lags=30, ax=axes[0, 1], color='#ef4444', method='ywm', title='PACF: Precios en Nivel (Lag 1 Dominante)')
plot_acf(btc['return_cop_1h'].dropna(), lags=30, ax=axes[1, 0], color='#10b981', title='ACF: Retornos Diferenciados I(0)')
plot_pacf(btc['return_cop_1h'].dropna(), lags=30, ax=axes[1, 1], color='#10b981', method='ywm', title='PACF: Retornos Diferenciados I(0)')

for ax in axes.flat:
    ax.set_ylim(-0.2, 1.05)
    ax.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig('figures/fig2_acf_pacf_comparison.png', dpi=300)
fig.savefig('public/figures/fig2_acf_pacf_comparison.png', dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# FIGURA 3: Volatilidad Condicional GARCH y Residuos Estandarizados
# -------------------------------------------------------------
print("[4/5] Generando Fig 3: GARCH(1,1) Volatilidad Condicional...")
train_val_end = int(len(btc) * 0.85)
am = arch_model(btc['return_cop_1h'].iloc[:train_val_end] * 100, vol='Garch', p=1, q=1, dist='studentst')
res_garch = am.fit(disp='off')

fig, (ax_vol, ax_res) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax_vol.plot(btc['timestamp'].iloc[:train_val_end], res_garch.conditional_volatility, color='#06b6d4', lw=1.2)
ax_vol.set_title('Volatilidad Condicional Estimada GARCH(1,1) con Distribución Student\'s t', fontsize=12, fontweight='bold')
ax_vol.set_ylabel('Sigma Condicional (%)', fontsize=10, fontweight='bold')

std_resid = res_garch.std_resid
ax_res.scatter(btc['timestamp'].iloc[:train_val_end], std_resid, color='#64748b', s=4, alpha=0.5)
ax_res.axhline(0, color='black', lw=0.8, ls='--')
ax_res.axhline(2, color='#ef4444', lw=0.8, ls=':')
ax_res.axhline(-2, color='#ef4444', lw=0.8, ls=':')
ax_res.set_title('Residuos Estandarizados (Verificación de Ruido Blanco y Colas Pesadas)', fontsize=12, fontweight='bold')
ax_res.set_ylabel('Residuo Std', fontsize=10, fontweight='bold')
ax_res.set_xlabel('Fecha / Hora', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig('figures/fig3_garch_volatility.png', dpi=300)
fig.savefig('public/figures/fig3_garch_volatility.png', dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# FIGURA 4: Desempeño en Test Reservado (Predicciones vs Realidad)
# -------------------------------------------------------------
print("[5/5] Generando Fig 4: Comparación Final de Modelos en Test...")
with open('public/model_results.json') as f:
    import json
    data_json = json.load(f)

cs = data_json['chart_series']
ts = pd.to_datetime(cs['timestamps'])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ts, np.array(cs['actual']) / 1e6, color='#0f172a', lw=2.5, label='Precio Real Test (BTC/COP en Millones)')
ax.plot(ts, np.array(cs['xgboost']) / 1e6, color='#10b981', lw=1.8, ls='-', label='XGBoost Optimizado (Adaptación Paper)')
ax.plot(ts, np.array(cs['sarimax']) / 1e6, color='#0284c7', lw=1.8, ls='--', label='SARIMAX(1,0,0) + Exógenas')
ax.plot(ts, np.array(cs['naive']) / 1e6, color='#f59e0b', lw=1.4, ls=':', label='Línea Base Naive (Persistencia)')

ax.set_title('Pronóstico Fuera de Muestra en Conjunto de Prueba Reservado (Test - 875 Horas)', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Precio (Millones COP)', fontsize=11, fontweight='bold')
ax.set_xlabel('Fecha y Hora de Prueba', fontsize=11, fontweight='bold')
ax.legend(loc='lower left', frameon=True, fontsize=10)

plt.tight_layout()
fig.savefig('figures/fig4_test_predictions_comparison.png', dpi=300)
fig.savefig('public/figures/fig4_test_predictions_comparison.png', dpi=300)
plt.close(fig)

# Copiar feature_importance si existe
import shutil
shutil.copy('data/feature_importance.png', 'figures/fig5_feature_importance.png')
shutil.copy('data/feature_importance.png', 'public/figures/fig5_feature_importance.png')

print("✓ Todas las figuras estáticas generadas exitosamente en 'figures/' y 'public/figures/'.")

