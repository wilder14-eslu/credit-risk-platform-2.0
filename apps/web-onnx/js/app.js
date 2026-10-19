/**
 * Credit Risk Platform 2.0 - Edge ONNX Web Application
 * Fast, reactive client-side application matching Streamlit with instant loading.
 */

document.addEventListener('DOMContentLoaded', async () => {
  const engine = new OnnxRiskEngine();
  let appData = null;
  let shapChartInstance = null;
  let monAucChartInstance = null;
  let monCalibChartInstance = null;
  let monDriftChartInstance = null;
  let modelsChartInstance = null;

  // Perfiles predefinidos idénticos a Streamlit
  const PROFILES = {
    conservative: {
      loan_amnt: 10000,
      term: 36,
      int_rate: 7.5,
      grade: "A",
      sub_grade: "A3",
      purpose: "credit_card",
      annual_inc: 95000,
      emp: 8,
      home: "MORTGAGE",
      verif: "Verified",
      dti: 9.0,
      state: "TX",
      fico: 760,
      revol_bal: 6000,
      revol_util: 22.0,
      open_acc: 9,
      total_acc: 28,
      history: 260
    },
    typical: {
      loan_amnt: 12000,
      term: 36,
      int_rate: 13.5,
      grade: "C",
      sub_grade: "C2",
      purpose: "debt_consolidation",
      annual_inc: 65000,
      emp: 5,
      home: "RENT",
      verif: "Source Verified",
      dti: 18.5,
      state: "CA",
      fico: 702,
      revol_bal: 14500,
      revol_util: 55.0,
      open_acc: 10,
      total_acc: 24,
      history: 180
    },
    risky: {
      loan_amnt: 30000,
      term: 60,
      int_rate: 24.0,
      grade: "F",
      sub_grade: "F2",
      purpose: "small_business",
      annual_inc: 38000,
      emp: 1,
      home: "RENT",
      verif: "Not Verified",
      dti: 34.0,
      state: "NV",
      fico: 662,
      revol_bal: 21000,
      revol_util: 92.0,
      open_acc: 15,
      total_acc: 18,
      history: 60
    }
  };

  // 1. Cargar datos estáticos precomputados
  try {
    const dataResp = await fetch('./data/app_data.json');
    appData = await dataResp.json();
    console.log('Datos de monitoreo y modelos cargados correctamente.');
  } catch (err) {
    console.warn('No se pudo cargar app_data.json:', err);
  }

  // 2. Inicializar ONNX Engine
  try {
    await engine.init();
    const statusDot = document.getElementById('engine-status');
    if (statusDot) {
      statusDot.innerHTML = '<span class="dot"></span> ONNX WASM Activo (< 3ms)';
    }
  } catch (err) {
    console.error('Error al inicializar ONNX Engine:', err);
  }

  // 3. Manejo de Navegación (Tabs)
  const navItems = document.querySelectorAll('.nav-item');
  const pages = document.querySelectorAll('.page');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetPageId = item.getAttribute('data-page');
      
      navItems.forEach(i => i.classList.remove('active'));
      pages.forEach(p => p.classList.remove('active'));

      item.classList.add('active');
      const targetPage = document.getElementById(targetPageId);
      if (targetPage) {
        targetPage.classList.add('active');
        // Renderizar charts específicos al entrar a la pestaña si aún no se han creado
        onTabActivated(targetPageId);
      }
    });
  });

  function onTabActivated(pageId) {
    if (pageId === 'page-monitoring') {
      renderMonitoringCharts();
    } else if (pageId === 'page-models') {
      renderModelsCharts();
    }
  }

  // 4. Formulario de Scoring y Sliders sync
  const form = document.getElementById('scoring-form');
  const sliderSyncs = [
    { slider: 'int_rate_slider', num: 'int_rate', valId: 'int_rate_val', suffix: '%' },
    { slider: 'dti_slider', num: 'dti', valId: 'dti_val', suffix: '%' },
    { slider: 'emp_slider', num: 'emp', valId: 'emp_val', suffix: ' años' },
    { slider: 'fico_slider', num: 'fico', valId: 'fico_val', suffix: '' },
    { slider: 'revol_util_slider', num: 'revol_util', valId: 'revol_util_val', suffix: '%' }
  ];

  sliderSyncs.forEach(s => {
    const sliderEl = document.getElementById(s.slider);
    const numEl = document.getElementById(s.num);
    const valEl = document.getElementById(s.valId);

    if (sliderEl && numEl) {
      sliderEl.addEventListener('input', (e) => {
        numEl.value = e.target.value;
        if (valEl) valEl.textContent = e.target.value + s.suffix;
      });
      numEl.addEventListener('input', (e) => {
        sliderEl.value = e.target.value;
        if (valEl) valEl.textContent = e.target.value + s.suffix;
      });
    }
  });

  // Grado y Subgrado dinámicos
  const gradeSelect = document.getElementById('grade');
  const subGradeSelect = document.getElementById('sub_grade');

  if (gradeSelect && subGradeSelect) {
    gradeSelect.addEventListener('change', () => {
      const g = gradeSelect.value;
      subGradeSelect.innerHTML = '';
      for (let i = 1; i <= 5; i++) {
        const opt = document.createElement('option');
        opt.value = `${g}${i}`;
        opt.textContent = `${g}${i}`;
        if (i === 2) opt.selected = true;
        subGradeSelect.appendChild(opt);
      }
    });
  }

  // Perfiles Switcher
  const pills = document.querySelectorAll('.pill[data-profile]');
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');

      const profileKey = pill.getAttribute('data-profile');
      if (PROFILES[profileKey]) {
        applyProfile(PROFILES[profileKey]);
        runEvaluation();
      }
    });
  });

  function applyProfile(p) {
    document.getElementById('loan_amnt').value = p.loan_amnt;
    document.getElementById('term').value = p.term;
    document.getElementById('int_rate').value = p.int_rate;
    document.getElementById('int_rate_slider').value = p.int_rate;
    document.getElementById('int_rate_val').textContent = p.int_rate + '%';

    document.getElementById('grade').value = p.grade;
    // Actualizar opciones de subgrado
    subGradeSelect.innerHTML = '';
    for (let i = 1; i <= 5; i++) {
      const opt = document.createElement('option');
      opt.value = `${p.grade}${i}`;
      opt.textContent = `${p.grade}${i}`;
      if (`${p.grade}${i}` === p.sub_grade) opt.selected = true;
      subGradeSelect.appendChild(opt);
    }

    document.getElementById('purpose').value = p.purpose;
    document.getElementById('annual_inc').value = p.annual_inc;
    document.getElementById('emp').value = p.emp;
    document.getElementById('emp_slider').value = p.emp;
    document.getElementById('emp_val').textContent = p.emp + ' años';

    document.getElementById('home').value = p.home;
    document.getElementById('verif').value = p.verif;
    document.getElementById('dti').value = p.dti;
    document.getElementById('dti_slider').value = p.dti;
    document.getElementById('dti_val').textContent = p.dti + '%';

    document.getElementById('state').value = p.state;
    document.getElementById('fico').value = p.fico;
    document.getElementById('fico_slider').value = p.fico;
    document.getElementById('fico_val').textContent = p.fico;

    document.getElementById('revol_bal').value = p.revol_bal;
    document.getElementById('revol_util').value = p.revol_util;
    document.getElementById('revol_util_slider').value = p.revol_util;
    document.getElementById('revol_util_val').textContent = p.revol_util + '%';

    document.getElementById('open_acc').value = p.open_acc;
    document.getElementById('total_acc').value = p.total_acc;
    document.getElementById('history').value = p.history;
  }

  // Capturar valores del formulario
  function getFormData() {
    return {
      loan_amnt: parseFloat(document.getElementById('loan_amnt').value) || 12000,
      term: parseInt(document.getElementById('term').value) || 36,
      int_rate: parseFloat(document.getElementById('int_rate').value) || 13.5,
      grade: document.getElementById('grade').value || "C",
      sub_grade: document.getElementById('sub_grade').value || "C2",
      purpose: document.getElementById('purpose').value || "debt_consolidation",
      annual_inc: parseFloat(document.getElementById('annual_inc').value) || 65000,
      emp: parseFloat(document.getElementById('emp').value) || 5,
      home: document.getElementById('home').value || "RENT",
      verif: document.getElementById('verif').value || "Source Verified",
      dti: parseFloat(document.getElementById('dti').value) || 18.5,
      state: document.getElementById('state').value || "CA",
      fico: parseFloat(document.getElementById('fico').value) || 702,
      revol_bal: parseFloat(document.getElementById('revol_bal').value) || 14500,
      revol_util: parseFloat(document.getElementById('revol_util').value) || 55,
      open_acc: parseFloat(document.getElementById('open_acc').value) || 10,
      total_acc: parseFloat(document.getElementById('total_acc').value) || 24,
      history: parseFloat(document.getElementById('history').value) || 180
    };
  }

  // Ejecución de inferencia instantánea
  async function runEvaluation() {
    const formData = getFormData();
    try {
      const res = await engine.predict(formData);
      renderScoringResults(res);
    } catch (err) {
      console.error('Error durante la inferencia ONNX:', err);
    }
  }

  const evalBtn = document.getElementById('btn-evaluar');
  if (evalBtn) {
    evalBtn.addEventListener('click', (e) => {
      e.preventDefault();
      runEvaluation();
    });
  }

  // Renderizar KPIs y Gráficos de Scoring
  function renderScoringResults(res) {
    const pdEl = document.getElementById('res-pd');
    const decisionEl = document.getElementById('res-decision');
    const bandEl = document.getElementById('res-band');
    const installmentEl = document.getElementById('res-installment');
    const latencyEl = document.getElementById('res-latency');

    if (pdEl) pdEl.textContent = `${(res.probability * 100).toFixed(1)}%`;
    if (decisionEl) {
      const isApproved = res.decision === "APROBAR";
      decisionEl.innerHTML = isApproved 
        ? `<span class="metric-badge badge-approved">✅ APROBAR</span>` 
        : `<span class="metric-badge badge-rejected">🛑 RECHAZAR</span>`;
    }
    if (bandEl) bandEl.textContent = `Banda ${res.risk_band}`;
    if (installmentEl) installmentEl.textContent = `$${Math.round(res.installment).toLocaleString()}`;
    if (latencyEl) latencyEl.textContent = `${res.latency_ms} ms (WASM Edge)`;

    // Gráfico de Factores SHAP
    renderShapChart(res.factors);

    // Tabla de factores
    const tableBody = document.querySelector('#factors-table tbody');
    if (tableBody) {
      tableBody.innerHTML = res.factors.map(f => {
        const isRisk = f.impact > 0;
        const color = isRisk ? '#f87171' : '#34d399';
        const labelText = isRisk ? 'Aumenta riesgo' : 'Reduce riesgo';
        return `
          <tr>
            <td style="font-weight: 500;">${f.label}</td>
            <td style="color: ${color}; font-family: monospace; font-weight: 600;">${f.impact > 0 ? '+' : ''}${f.impact.toFixed(3)}</td>
            <td><span class="status-pill ${isRisk ? 'status-alerta' : 'status-estable'}">${labelText}</span></td>
          </tr>
        `;
      }).join('');
    }
  }

  function renderShapChart(factors) {
    const canvas = document.getElementById('shapChart');
    if (!canvas) return;

    // Ordenar de menor a mayor impacto para barras horizontales
    const sorted = [...factors].sort((a, b) => a.impact - b.impact);
    const labels = sorted.map(f => f.label);
    const dataVals = sorted.map(f => f.impact);
    const bgColors = sorted.map(f => f.impact > 0 ? '#ef4444' : '#10b981');

    if (shapChartInstance) {
      shapChartInstance.destroy();
    }

    const ctx = canvas.getContext('2d');
    shapChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          data: dataVals,
          backgroundColor: bgColors,
          borderRadius: 4,
          borderSkipped: false
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
              label: (ctx) => ` Contribución log-odds: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(3)}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            ticks: { color: '#94a3b8' },
            title: {
              display: true,
              text: 'Contribución al log-odds de default (SHAP)',
              color: '#94a3b8'
            }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#e2e8f0', font: { weight: 500 } }
          }
        }
      }
    });
  }

  // 5. Monitoreo Charts & Tables
  let monitoringRendered = false;
  function renderMonitoringCharts() {
    if (monitoringRendered || !appData || !appData.monitoring) return;
    monitoringRendered = true;

    const mon = appData.monitoring.metrics;
    const months = mon.map(m => m.clock_month);
    const aucVals = mon.map(m => m.roc_auc);
    const obsVals = mon.map(m => m.default_rate_observed * 100);
    const predVals = mon.map(m => m.default_rate_predicted * 100);

    // Chart 1: AUC en Producción
    const aucCanvas = document.getElementById('monAucChart');
    if (aucCanvas) {
      monAucChartInstance = new Chart(aucCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: months,
          datasets: [{
            label: 'ROC-AUC en producción',
            data: aucVals,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.2,
            pointRadius: 4,
            pointBackgroundColor: '#3b82f6'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { mode: 'index', intersect: false }
          },
          scales: {
            y: {
              min: 0.67,
              max: 0.71,
              grid: { color: 'rgba(255, 255, 255, 0.08)' },
              ticks: { color: '#94a3b8' },
              title: { display: true, text: 'ROC-AUC', color: '#94a3b8' }
            },
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8', maxTicksLimit: 12 }
            }
          }
        }
      });
    }

    // Chart 2: Calibración Observada vs Predicha
    const calibCanvas = document.getElementById('monCalibChart');
    if (calibCanvas) {
      monCalibChartInstance = new Chart(calibCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: months,
          datasets: [
            {
              label: 'Default observado',
              data: obsVals,
              borderColor: '#3b82f6',
              tension: 0.2,
              pointRadius: 3
            },
            {
              label: 'Default predicho',
              data: predVals,
              borderColor: '#f97316',
              borderDash: [5, 5],
              tension: 0.2,
              pointRadius: 3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#e2e8f0' } }
          },
          scales: {
            y: {
              grid: { color: 'rgba(255, 255, 255, 0.08)' },
              ticks: {
                color: '#94a3b8',
                callback: val => `${val.toFixed(0)}%`
              },
              title: { display: true, text: 'Tasa de Default', color: '#94a3b8' }
            },
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8', maxTicksLimit: 12 }
            }
          }
        }
      });
    }

    // Chart 3: PSI Data Drift Horizontal
    const driftData = appData.monitoring.drift;
    const driftCanvas = document.getElementById('monDriftChart');
    if (driftCanvas) {
      monDriftChartInstance = new Chart(driftCanvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels: driftData.map(d => d.feature),
          datasets: [{
            data: driftData.map(d => d.psi),
            backgroundColor: driftData.map(d => d.status === 'warning' ? '#f59e0b' : (d.status === 'alerta' ? '#ef4444' : '#10b981')),
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
                label: (ctx) => ` PSI: ${ctx.raw.toFixed(3)} (${driftData[ctx.dataIndex].status})`
              }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.08)' },
              ticks: { color: '#94a3b8' },
              title: { display: true, text: 'Population Stability Index (PSI)', color: '#94a3b8' }
            },
            y: {
              grid: { display: false },
              ticks: { color: '#e2e8f0' }
            }
          }
        }
      });
    }

    // Tablas de Monitoreo
    const driftTableBody = document.querySelector('#drift-table tbody');
    if (driftTableBody) {
      driftTableBody.innerHTML = driftData.map(d => `
        <tr>
          <td><code>${d.feature}</code></td>
          <td>${d.kind}</td>
          <td><b>${d.psi.toFixed(3)}</b></td>
          <td>${d.ks_pvalue !== null ? d.ks_pvalue.toFixed(3) : '-'}</td>
          <td><span class="status-pill status-${d.status}">${d.status}</span></td>
        </tr>
      `).join('');
    }

    const abTableBody = document.querySelector('#ab-table tbody');
    if (abTableBody) {
      abTableBody.innerHTML = appData.monitoring.ab_test.map(a => `
        <tr>
          <td>${a.run_ts}</td>
          <td>${a.champion_version}</td>
          <td>${a.challenger_version}</td>
          <td>${a.champion_auc.toFixed(4)}</td>
          <td>${a.challenger_auc.toFixed(4)}</td>
          <td>${a.p_value.toFixed(3)}</td>
          <td>${a.decision}</td>
        </tr>
      `).join('');
    }

    const retrainTableBody = document.querySelector('#retrain-table tbody');
    if (retrainTableBody) {
      retrainTableBody.innerHTML = appData.monitoring.retrains.map(r => `
        <tr>
          <td>${r.event_ts}</td>
          <td>${r.trigger_reason}</td>
          <td>${r.previous_version}</td>
          <td><b>${r.new_version}</b></td>
          <td>${r.result}</td>
        </tr>
      `).join('');
    }
  }

  // 6. Modelos Charts & Tables
  let modelsRendered = false;
  function renderModelsCharts() {
    if (modelsRendered || !appData) return;
    modelsRendered = true;

    const bench = appData.results.benchmark;
    const benchTableBody = document.querySelector('#benchmark-table tbody');
    if (benchTableBody) {
      benchTableBody.innerHTML = bench.map(b => `
        <tr>
          <td><b>${b.algorithm}</b></td>
          <td>${b.train_auc.toFixed(4)}</td>
          <td>${b.val_auc.toFixed(4)}</td>
          <td><b>${b.test_auc.toFixed(4)}</b></td>
          <td>${(b.val_gap * 100).toFixed(2)}%</td>
          <td>${b.test_ks.toFixed(3)}</td>
          <td>${b.test_brier.toFixed(4)}</td>
          <td>${b.latency_ms.toFixed(3)} ms</td>
        </tr>
      `).join('');
    }

    // Chart DeLong IC 95%
    const evRows = appData.monitoring.evaluation.filter(e => e.split === 'validation');
    const modelsCanvas = document.getElementById('modelsDeLongChart');
    if (modelsCanvas && evRows.length > 0) {
      modelsChartInstance = new Chart(modelsCanvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels: evRows.map(r => r.model),
          datasets: [{
            data: evRows.map(r => r.auc),
            backgroundColor: evRows.map(r => r.selected ? '#3b82f6' : '#64748b'),
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
                label: (ctx) => ` AUC: ${ctx.raw.toFixed(4)} (IC 95%: [${evRows[ctx.dataIndex].auc_ci_low.toFixed(4)}, ${evRows[ctx.dataIndex].auc_ci_high.toFixed(4)}])`
              }
            }
          },
          scales: {
            x: {
              min: 0.67,
              max: 0.71,
              grid: { color: 'rgba(255, 255, 255, 0.08)' },
              ticks: { color: '#94a3b8' }
            },
            y: {
              grid: { display: false },
              ticks: { color: '#e2e8f0' }
            }
          }
        }
      });
    }

    // Modal para visor de figuras
    setupFigureModal();
  }

  function setupFigureModal() {
    const modal = document.getElementById('figModal');
    const modalImg = document.getElementById('modalImg');
    const modalClose = document.getElementById('modalClose');

    document.querySelectorAll('.figure-img-wrap img').forEach(img => {
      img.addEventListener('click', () => {
        if (modal && modalImg) {
          modalImg.src = img.src;
          modal.classList.add('active');
        }
      });
    });

    if (modalClose && modal) {
      modalClose.addEventListener('click', () => {
        modal.classList.remove('active');
      });
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
      });
    }
  }

  // Iniciar evaluación del perfil típico por defecto
  applyProfile(PROFILES.typical);
  runEvaluation();
});
