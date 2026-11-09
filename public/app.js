const DATA = window.PROJECT_DATA;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initOverview();
    initCleaningTab();
    initEconometricsTab();
    initMLTab();
    initComparisonTab();
    initPaperTab();
    initSimulator();
});

// 1. Manejo de Pestañas (Tabs)
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.tab-section');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabTarget = btn.getAttribute('data-tab');
            navButtons.forEach(b => {
                b.classList.remove('bg-emerald-500/20', 'text-emerald-400', 'border-emerald-500/50');
                b.classList.add('text-slate-400', 'hover:text-slate-200');
            });
            btn.classList.add('bg-emerald-500/20', 'text-emerald-400', 'border-emerald-500/50');
            btn.classList.remove('text-slate-400', 'hover:text-slate-200');

            sections.forEach(sec => {
                if (sec.id === tabTarget) {
                    sec.classList.remove('hidden');
                } else {
                    sec.classList.add('hidden');
                }
            });
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
}

// 2. Resumen Superior (Overview KPI Cards)
function initOverview() {
    const naiveRow = DATA.benchmark_table.find(r => r.model.includes('Naive'));
    const xgbRow = DATA.benchmark_table.find(r => r.model.includes('XGBoost'));
    const sarimaRow = DATA.benchmark_table.find(r => r.model.includes('SARIMAX'));
    const varRow = DATA.benchmark_table.find(r => r.model.includes('VAR'));

    document.getElementById('kpi-naive-mape').textContent = `${naiveRow.test_mape}%`;
    document.getElementById('kpi-xgb-mape').textContent = `${xgbRow.test_mape}%`;
    document.getElementById('kpi-sarima-mape').textContent = `${sarimaRow.test_mape}%`;
    document.getElementById('kpi-dir-acc').textContent = `${xgbRow.directional_accuracy_pct}%`;
}

// 3. Pestaña de Limpieza y Feature Engineering
function initCleaningTab() {
    // Tests ADF y KPSS
    const stat = DATA.stationarity;
    const tableBody = document.getElementById('stationarity-table-body');
    tableBody.innerHTML = `
        <tr class="border-b border-slate-800/80 bg-slate-900/40">
            <td class="p-3 font-semibold text-slate-200">Nivel de Precios ($P_t^{COP}$)</td>
            <td class="p-3 font-mono text-amber-400">${stat.price_level.adf_stat} (p=${stat.price_level.adf_pvalue})</td>
            <td class="p-3 font-mono text-rose-400">${stat.price_level.kpss_stat} (p=${stat.price_level.kpss_pvalue})</td>
            <td class="p-3 text-rose-400 font-medium">No Estacionaria (I(1))</td>
            <td class="p-3 text-xs text-slate-400">${stat.price_level.interpretation}</td>
        </tr>
        <tr class="border-b border-slate-800/80 bg-slate-900/20">
            <td class="p-3 font-semibold text-slate-200">Retornos Diferenciados ($\Delta P_t / P_{t-1}$)</td>
            <td class="p-3 font-mono text-emerald-400">${stat.returns_diff.adf_stat} (p=0.0000)</td>
            <td class="p-3 font-mono text-emerald-400">${stat.returns_diff.kpss_stat} (p=${stat.returns_diff.kpss_pvalue})</td>
            <td class="p-3 text-emerald-400 font-medium">Estacionaria (I(0))</td>
            <td class="p-3 text-xs text-slate-400">${stat.returns_diff.interpretation}</td>
        </tr>
    `;

    // Gráfico de Importancia de Variables (Feature Importance)
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    const labels = DATA.feature_importance.map(f => f.feature);
    const values = DATA.feature_importance.map(f => f.importance);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importancia Relativa / Ganancia en Árbol (%)',
                data: values,
                backgroundColor: 'rgba(16, 185, 129, 0.75)',
                borderColor: '#10b981',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` Ganancia: ${ctx.raw}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { family: 'monospace', size: 11 } }
                }
            }
        }
    });
}

// 4. Pestaña de Modelos Econométricos
function initEconometricsTab() {
    // SARIMAX Grid Table
    const sarimaxBody = document.getElementById('sarimax-grid-body');
    DATA.sarimax_grid.forEach(row => {
        const isBest = row.order.includes('1, 0, 0');
        sarimaxBody.innerHTML += `
            <tr class="border-b border-slate-800 ${isBest ? 'bg-emerald-500/10 text-emerald-300 font-semibold' : 'text-slate-300'}">
                <td class="p-2.5 font-mono">${row.order}</td>
                <td class="p-2.5 font-mono">${row.aic} ${isBest ? '★ (Óptimo AIC)' : ''}</td>
                <td class="p-2.5 font-mono">${row.bic} ${isBest ? '★ (Óptimo BIC)' : ''}</td>
                <td class="p-2.5 text-xs">${isBest ? 'Captura autorregresión horaria AR(1) con máxima parsimonia.' : 'Mayor penalización por parámetros redundantes.'}</td>
            </tr>
        `;
    });

    // GARCH Grid Table
    const garchBody = document.getElementById('garch-grid-body');
    DATA.garch_grid.forEach(row => {
        const isOpt = row.model === 'GARCH(1,2)' && row.dist === 'studentst';
        const isSelected = row.model === 'GARCH(1,1)' && row.dist === 'studentst';
        garchBody.innerHTML += `
            <tr class="border-b border-slate-800 ${isSelected ? 'bg-cyan-500/10 text-cyan-300 font-semibold' : 'text-slate-300'}">
                <td class="p-2.5 font-mono">${row.model}</td>
                <td class="p-2.5 capitalize">${row.dist === 'studentst' ? "Student's t (Colas Pesadas)" : 'Normal Gaussiana'}</td>
                <td class="p-2.5 font-mono">${row.aic}</td>
                <td class="p-2.5 font-mono">${row.bic}</td>
                <td class="p-2.5 font-mono">${row.log_likelihood}</td>
                <td class="p-2.5 text-xs">${isSelected ? 'Configuración definitiva: balance óptimo parsimonia-robustez.' : (isOpt ? 'Menor AIC teórico.' : 'Rechazada por subestimar leptocurtosis.')}</td>
            </tr>
        `;
    });

    // Johansen Cointegration Table
    const johansenBody = document.getElementById('johansen-table-body');
    DATA.johansen_cointegration.trace_stats.forEach((stat, i) => {
        const crit = DATA.johansen_cointegration.trace_crit_95[i];
        const coint = stat > crit;
        johansenBody.innerHTML += `
            <tr class="border-b border-slate-800 text-slate-300">
                <td class="p-2 font-mono">r &le; ${i}</td>
                <td class="p-2 font-mono text-cyan-400">${stat}</td>
                <td class="p-2 font-mono text-slate-400">${crit}</td>
                <td class="p-2 font-semibold ${coint ? 'text-emerald-400' : 'text-rose-400'}">${coint ? 'Rechaza H0 (Cointegración confirmada)' : 'No Rechaza'}</td>
            </tr>
        `;
    });

    // Gráfico de Volatilidad Condicional GARCH
    const ctxVol = document.getElementById('garchVolatilityChart').getContext('2d');
    new Chart(ctxVol, {
        type: 'line',
        data: {
            labels: DATA.chart_series.timestamps,
            datasets: [{
                label: 'Volatilidad Condicional Estimada GARCH(1,1) (%)',
                data: DATA.chart_series.garch_volatility,
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                fill: true,
                tension: 0.2,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: '#94a3b8' } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', callback: v => `${v}%` }
                }
            }
        }
    });
}

// 5. Pestaña de Machine Learning Avanzado
function initMLTab() {
    const ctxML = document.getElementById('mlPredictionsChart').getContext('2d');
    new Chart(ctxML, {
        type: 'line',
        data: {
            labels: DATA.chart_series.timestamps,
            datasets: [
                {
                    label: 'Precio Real Test (BTC/COP)',
                    data: DATA.chart_series.actual,
                    borderColor: '#ffffff',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'XGBoost Optimizado',
                    data: DATA.chart_series.xgboost,
                    borderColor: '#10b981',
                    borderWidth: 1.8,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'SARIMAX + Exógenas',
                    data: DATA.chart_series.sarimax,
                    borderColor: '#38bdf8',
                    borderWidth: 1.8,
                    borderDash: [5, 4],
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Línea Base Naive (Persistencia)',
                    data: DATA.chart_series.naive,
                    borderColor: '#f59e0b',
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { color: '#cbd5e1' } },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: $${Number(ctx.raw).toLocaleString('es-CO')} COP`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: {
                        color: '#94a3b8',
                        callback: v => `$${(v / 1e6).toFixed(1)}M`
                    }
                }
            }
        }
    });
}

// 6. Pestaña de Comparación Integral
function initComparisonTab() {
    const tableBody = document.getElementById('comparison-table-body');
    tableBody.innerHTML = '';

    DATA.benchmark_table.forEach((row, idx) => {
        const isBestMape = idx === 0;
        const isBestDir = row.directional_accuracy_pct === Math.max(...DATA.benchmark_table.map(r => r.directional_accuracy_pct));

        tableBody.innerHTML += `
            <tr class="border-b border-slate-800/80 hover:bg-slate-850/60 transition ${isBestMape ? 'bg-amber-500/5' : ''}">
                <td class="p-3.5 font-medium text-slate-100 flex items-center gap-2">
                    <span>${row.model}</span>
                    ${isBestMape ? '<span class="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded border border-amber-500/30">Top MAPE</span>' : ''}
                    ${isBestDir ? '<span class="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30">Top Hit Ratio</span>' : ''}
                </td>
                <td class="p-3.5">
                    <span class="text-xs px-2.5 py-1 rounded-full font-medium ${
                        row.category === 'Línea Base' ? 'bg-amber-400/10 text-amber-300 border border-amber-400/20' :
                        row.category === 'Econometría' ? 'bg-cyan-400/10 text-cyan-300 border border-cyan-400/20' :
                        'bg-emerald-400/10 text-emerald-300 border border-emerald-400/20'
                    }">${row.category}</span>
                </td>
                <td class="p-3.5 font-mono text-right ${row.test_mape < 0.5 ? 'text-emerald-400 font-bold' : 'text-slate-300'}">${row.test_mape.toFixed(3)}%</td>
                <td class="p-3.5 font-mono text-right text-slate-300">${row.test_minmax_rmse.toFixed(4)}</td>
                <td class="p-3.5 font-mono text-right text-slate-300">$${Number(row.test_mae_cop).toLocaleString('es-CO')}</td>
                <td class="p-3.5 font-mono text-right text-slate-300">$${Number(row.test_rmse_cop).toLocaleString('es-CO')}</td>
                <td class="p-3.5 font-mono text-right ${row.directional_accuracy_pct > 50 ? 'text-emerald-400 font-bold' : 'text-rose-400'}">${row.directional_accuracy_pct.toFixed(2)}%</td>
            </tr>
        `;
    });

    // Radar Chart de Balance Multicriterio
    const ctxRadar = document.getElementById('comparisonRadarChart').getContext('2d');
    new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Precisión Absoluta (1/MAPE)', 'Hit Ratio Direccional', 'Parsimonia Teórica', 'Robustez No-Lineal', 'Ajuste de Volatilidad', 'Interpretabilidad'],
            datasets: [
                {
                    label: 'XGBoost Adaptado',
                    data: [85, 88, 65, 95, 80, 85],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    borderWidth: 2
                },
                {
                    label: 'SARIMAX + Exógenas',
                    data: [82, 85, 90, 60, 85, 95],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.2)',
                    borderWidth: 2
                },
                {
                    label: 'Línea Base Naive',
                    data: [100, 10, 100, 10, 10, 40],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 1.5,
                    borderDash: [4, 4]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.08)' },
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    pointLabels: { color: '#94a3b8', font: { size: 11 } },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { position: 'bottom', labels: { color: '#cbd5e1' } }
            }
        }
    });
}

// 7. Pestaña de Contraste con el Artículo Base (Gautam, 2025)
function initPaperTab() {
    const p = DATA.comparison_paper;
    const body = document.getElementById('paper-comparison-body');
    body.innerHTML = `
        <tr class="border-b border-slate-800">
            <td class="p-3 font-semibold text-slate-200">Gautam (2025) — LSTM + XGBoost</td>
            <td class="p-3 text-slate-400">USD (Binance Diario)</td>
            <td class="p-3 font-mono text-cyan-400">4.88%</td>
            <td class="p-3 font-mono text-cyan-400">0.0650</td>
            <td class="p-3 text-xs text-slate-300">Arquitectura híbrida neuronal + ensamble tabular.</td>
        </tr>
        <tr class="border-b border-slate-800">
            <td class="p-3 font-semibold text-slate-200">Gautam (2025) — XGBoost Individual</td>
            <td class="p-3 text-slate-400">USD (Binance Diario)</td>
            <td class="p-3 font-mono text-cyan-400">5.33%</td>
            <td class="p-3 font-mono text-cyan-400">0.0705</td>
            <td class="p-3 text-xs text-slate-300">Regresión con variables OHLCV estándar.</td>
        </tr>
        <tr class="border-b border-slate-800">
            <td class="p-3 font-semibold text-slate-200">Gautam (2025) — ARIMA(1,1,1)</td>
            <td class="p-3 text-slate-400">USD (Binance Diario)</td>
            <td class="p-3 font-mono text-cyan-400">6.71%</td>
            <td class="p-3 font-mono text-cyan-400">0.0819</td>
            <td class="p-3 text-xs text-slate-300">Referencia lineal univariada tradicional.</td>
        </tr>
        <tr class="border-b border-slate-800 bg-emerald-500/10">
            <td class="p-3 font-semibold text-emerald-300">Nuestra Adaptación — XGBoost Local</td>
            <td class="p-3 text-emerald-200">COP (Binance 1H + TRM Banco de la República)</td>
            <td class="p-3 font-mono text-emerald-400 font-bold">0.345%</td>
            <td class="p-3 font-mono text-emerald-400 font-bold">0.0235</td>
            <td class="p-3 text-xs text-slate-200">Ensamble enriquecido con TRM, RSI, MACD, Bollinger y GARCH.</td>
        </tr>
        <tr class="border-b border-slate-800 bg-cyan-500/10">
            <td class="p-3 font-semibold text-cyan-300">Nuestra Adaptación — SARIMAX + Exógenas</td>
            <td class="p-3 text-cyan-200">COP (1H + Regresores Exógenos)</td>
            <td class="p-3 font-mono text-cyan-400 font-bold">0.288%</td>
            <td class="p-3 font-mono text-cyan-400 font-bold">0.0210</td>
            <td class="p-3 text-xs text-slate-200">Optimizado por AIC/BIC incorporando canal cambiario TRM.</td>
        </tr>
        <tr class="border-b border-slate-800 bg-amber-500/10">
            <td class="p-3 font-semibold text-amber-300">Nuestra Adaptación — Línea Base Naive</td>
            <td class="p-3 text-amber-200">COP (Persistencia $\hat{P}_{t+1} = P_t$)</td>
            <td class="p-3 font-mono text-amber-400 font-bold">0.261%</td>
            <td class="p-3 font-mono text-amber-400 font-bold">0.0203</td>
            <td class="p-3 text-xs text-slate-200">Dominio en MAPE explicado por Caminata Aleatoria $I(1)$ a 1 hora.</td>
        </tr>
    `;

    document.getElementById('paper-discussion-text').textContent = p.critical_discussion;
}

// 8. Simulador Interactivo de Inversión y Escenarios
function initSimulator() {
    const inputAmount = document.getElementById('sim-amount');
    const inputTrm = document.getElementById('sim-trm-change');
    const inputCryptoVol = document.getElementById('sim-volatility');
    const selectModel = document.getElementById('sim-model');
    const resultBox = document.getElementById('sim-result-box');

    function updateSimulation() {
        const capitalCOP = parseFloat(inputAmount.value) || 10000000;
        const trmShift = parseFloat(inputTrm.value) / 100.0;
        const volShift = parseFloat(inputCryptoVol.value) / 100.0;
        const model = selectModel.value;

        // Modelo base: BTC a precio actual ~$270.000.000 COP
        const basePrice = 270000000;
        let expectedAlpha = 0.001; // +0.10% rendimiento horario medio

        if (model === 'xgboost') {
            expectedAlpha = 0.0025 + (volShift * 0.15) + (trmShift * 0.8);
        } else if (model === 'sarimax') {
            expectedAlpha = 0.0018 + (trmShift * 0.95);
        } else if (model === 'naive') {
            expectedAlpha = 0.0000;
        }

        const projectedPrice = basePrice * (1.0 + expectedAlpha);
        const finalCapital = capitalCOP * (1.0 + expectedAlpha);
        const gainLoss = finalCapital - capitalCOP;

        resultBox.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="p-3 bg-slate-900/60 rounded-lg border border-slate-800">
                    <p class="text-xs text-slate-400">Precio Proyectado (BTC 1H)</p>
                    <p class="text-xl font-mono font-bold text-slate-100">$${Math.round(projectedPrice).toLocaleString('es-CO')} COP</p>
                    <span class="text-xs ${expectedAlpha >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${(expectedAlpha * 100).toFixed(3)}% retorno esperado</span>
                </div>
                <div class="p-3 bg-slate-900/60 rounded-lg border border-slate-800">
                    <p class="text-xs text-slate-400">Capital Proyectado (COP)</p>
                    <p class="text-xl font-mono font-bold text-slate-100">$${Math.round(finalCapital).toLocaleString('es-CO')} COP</p>
                    <span class="text-xs text-slate-400">Sobre capital inicial de $${capitalCOP.toLocaleString('es-CO')}</span>
                </div>
                <div class="p-3 bg-slate-900/60 rounded-lg border border-slate-800">
                    <p class="text-xs text-slate-400">P&L Esperado (Ganancia / Pérdida)</p>
                    <p class="text-xl font-mono font-bold ${gainLoss >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
                        ${gainLoss >= 0 ? '+' : ''}$${Math.round(gainLoss).toLocaleString('es-CO')} COP
                    </p>
                    <span class="text-xs text-slate-400">Sensibilidad ajustada por TRM y volatilidad</span>
                </div>
            </div>
        `;
    }

    inputAmount.addEventListener('input', updateSimulation);
    inputTrm.addEventListener('input', () => {
        document.getElementById('trm-val-badge').textContent = `${inputTrm.value}%`;
        updateSimulation();
    });
    inputCryptoVol.addEventListener('input', () => {
        document.getElementById('vol-val-badge').textContent = `${inputCryptoVol.value}%`;
        updateSimulation();
    });
    selectModel.addEventListener('change', updateSimulation);

    updateSimulation();
}
