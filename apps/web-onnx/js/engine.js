/**
 * ONNX Edge Inference Engine for Credit Risk Scoring (WASM In-Browser Execution)
 * Replicates Python Preprocessor and build_features with zero server roundtrips.
 */

class OnnxRiskEngine {
  constructor() {
    this.session = null;
    this.meta = null;
    this.isReady = false;
    this.initPromise = null;
  }

  async init(metaPath = './model/credit_risk_meta.json', modelPath = './model/credit_risk_model.onnx') {
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      console.log('⚡ Inicializando ONNX Edge Engine...');
      // Configurar rutas para los binarios WASM de ONNX Runtime
      if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
        ort.env.wasm.wasmPaths = './js/';
        ort.env.wasm.numThreads = 1; // Máxima compatibilidad cross-origin
      }

      // 1. Cargar metadatos
      const metaResp = await fetch(metaPath);
      this.meta = await metaResp.json();
      console.log('Metadatos cargados. Columnas requeridas:', this.meta.feature_columns.length);

      // 2. Inicializar sesión ONNX Runtime Web
      if (typeof ort === 'undefined') {
        throw new Error('onnxruntime-web (ort) no está cargado en el contexto global.');
      }

      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });

      this.isReady = true;
      console.log('✅ Sesión ONNX cargada exitosamente en WebAssembly.');
      return this;
    })();

    return this.initPromise;
  }

  calculateInstallment(loanAmnt, intRate, term) {
    const r = intRate / 1200.0;
    if (r === 0) return loanAmnt / term;
    return (loanAmnt * r) / (1.0 - Math.pow(1.0 + r, -term));
  }

  riskBand(probability) {
    if (probability < 0.05) return "A";
    if (probability < 0.10) return "B";
    if (probability < 0.20) return "C";
    if (probability < 0.35) return "D";
    return "E";
  }

  preprocess(form) {
    const loan = parseFloat(form.loan_amnt) || 12000.0;
    const term = parseInt(form.term) || 36;
    const intRate = parseFloat(form.int_rate) || 13.5;
    const annualInc = parseFloat(form.annual_inc) || 65000.0;
    const emp = parseFloat(form.emp) || 5.0;
    const dti = parseFloat(form.dti) || 18.5;
    const fico = parseFloat(form.fico) || 702.0;
    const revolBal = parseFloat(form.revol_bal) || 14500.0;
    const revolUtil = parseFloat(form.revol_util) || 55.0;
    const openAcc = parseFloat(form.open_acc) || 10.0;
    const totalAcc = parseFloat(form.total_acc) || 24.0;
    const history = parseFloat(form.history) || 180.0;
    const home = form.home || "RENT";
    const verif = form.verif || "Source Verified";
    const purpose = form.purpose || "debt_consolidation";
    const state = (form.state || "CA").trim().toUpperCase().substring(0, 2);
    const grade = form.grade || "C";
    const subGrade = form.sub_grade || "C2";

    const inst = this.calculateInstallment(loan, intRate, term);
    const medians = this.meta.medians || {};

    // Diccionario de variables numéricas y derivadas
    const row = {
      loan_amnt: loan,
      term_months: term,
      int_rate: intRate,
      installment: inst,
      emp_length_years: emp,
      annual_inc: annualInc,
      dti: dti,
      delinq_2yrs: 0.0,
      fico_score: fico,
      inq_last_6mths: 1.0,
      open_acc: openAcc,
      pub_rec: 0.0,
      revol_bal: revolBal,
      revol_util: revolUtil,
      total_acc: totalAcc,
      mort_acc: home === "MORTGAGE" ? 1.0 : 0.0,
      pub_rec_bankruptcies: 0.0,
      credit_history_months: history,
      loan_to_income: annualInc > 0 ? loan / annualInc : (medians.loan_to_income || 0.18),
      installment_to_income: annualInc > 0 ? (inst * 12.0) / annualInc : (medians.installment_to_income || 0.06),
      revol_bal_to_income: annualInc > 0 ? revolBal / annualInc : (medians.revol_bal_to_income || 0.22),
      open_acc_ratio: totalAcc > 0 ? openAcc / totalAcc : (medians.open_acc_ratio || 0.5),
      log_annual_inc: Math.log1p(Math.max(0, annualInc)),
      "emp_length_years=__missing__": 0.0
    };

    // Variables categóricas
    const cats = {
      grade: grade,
      sub_grade: subGrade,
      home_ownership: home,
      verification_status: verif,
      purpose: purpose,
      addr_state: state,
      application_type: "Individual",
      initial_list_status: "w"
    };

    const vocab = this.meta.vocab || {};
    for (const [col, categories] of Object.entries(vocab)) {
      const val = cats[col] || "";
      const isKnown = categories.includes(val);

      for (const cat of categories) {
        row[`${col}=${cat}`] = (val === cat) ? 1.0 : 0.0;
      }
      row[`${col}=__other__`] = isKnown ? 0.0 : 1.0;
    }

    // Vector ordenado de 133 dimensiones
    const vec = new Float32Array(this.meta.feature_columns.length);
    for (let i = 0; i < this.meta.feature_columns.length; i++) {
      const colName = this.meta.feature_columns[i];
      vec[i] = (row[colName] !== undefined) ? row[colName] : (medians[colName] || 0.0);
    }

    return { vec, inst };
  }

  computeExplainFactors(form, probability, installment) {
    const fico = parseFloat(form.fico) || 702;
    const dti = parseFloat(form.dti) || 18.5;
    const intRate = parseFloat(form.int_rate) || 13.5;
    const inc = parseFloat(form.annual_inc) || 65000;
    const grade = form.grade || "C";
    const subGrade = form.sub_grade || "C2";
    const revolUtil = parseFloat(form.revol_util) || 55;
    const history = parseFloat(form.history) || 180;
    const loan = parseFloat(form.loan_amnt) || 12000;

    // Aproximación analítica de SHAP centrada en la población de entrenamiento
    const factorsList = [];

    // 1. Grado / Subgrado (fuerte peso log-odds)
    const gradeIndex = "ABCDEFG".indexOf(grade.toUpperCase());
    const subNum = parseInt(subGrade.slice(1)) || 3;
    const gradeVal = (gradeIndex !== -1 ? gradeIndex * 5 + subNum : 13) - 13;
    const gradeImpact = gradeVal * 0.055;
    factorsList.push({
      feature: "sub_grade",
      label: `Subgrado (${subGrade})`,
      impact: gradeImpact
    });

    // 2. FICO Score (baseline 700)
    const ficoImpact = ((700 - fico) / 50.0) * 0.16;
    factorsList.push({
      feature: "fico_score",
      label: `Score FICO (${fico})`,
      impact: ficoImpact
    });

    // 3. Tasa de interés (baseline 13.5%)
    const rateImpact = ((intRate - 13.5) / 5.0) * 0.14;
    factorsList.push({
      feature: "int_rate",
      label: `Tasa de interés (${intRate}%)`,
      impact: rateImpact
    });

    // 4. DTI (baseline 18%)
    const dtiImpact = ((dti - 18.0) / 10.0) * 0.12;
    factorsList.push({
      feature: "dti",
      label: `Deuda / ingreso, DTI (${dti}%)`,
      impact: dtiImpact
    });

    // 5. Ingreso anual (baseline 65,000 USD)
    const logInc = Math.log1p(Math.max(1, inc));
    const logBaseline = Math.log1p(65000);
    const incImpact = -(logInc - logBaseline) * 0.15;
    factorsList.push({
      feature: "annual_inc",
      label: `Ingreso anual ($${Math.round(inc).toLocaleString()})`,
      impact: incImpact
    });

    // 6. Utilización revolvente (baseline 55%)
    const utilImpact = ((revolUtil - 55.0) / 30.0) * 0.08;
    factorsList.push({
      feature: "revol_util",
      label: `Utilización revolvente (${revolUtil}%)`,
      impact: utilImpact
    });

    // 7. Monto / Cuota (loan_to_income)
    const lti = inc > 0 ? loan / inc : 0.2;
    const ltiImpact = (lti - 0.18) * 0.45;
    factorsList.push({
      feature: "loan_to_income",
      label: `Monto / Ingreso (${(lti * 100).toFixed(1)}%)`,
      impact: ltiImpact
    });

    // 8. Antigüedad crediticia
    const histImpact = -((history - 180) / 120.0) * 0.05;
    factorsList.push({
      feature: "credit_history_months",
      label: `Antigüedad crediticia (${history} m)`,
      impact: histImpact
    });

    // Ordenar por magnitud absoluta de impacto
    factorsList.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

    // Tomar los top 6 y asignar dirección
    return factorsList.slice(0, 6).map(f => ({
      ...f,
      impact: parseFloat(f.impact.toFixed(3)),
      direction: f.impact > 0 ? "aumenta_riesgo" : "reduce_riesgo"
    }));
  }

  async predict(form) {
    if (!this.isReady) {
      await this.init();
    }

    const tStart = performance.now();
    const { vec, inst } = this.preprocess(form);

    const tensorInput = new ort.Tensor('float32', vec, [1, this.meta.feature_columns.length]);
    const feeds = { float_input: tensorInput };

    const output = await this.session.run(feeds);
    const tEnd = performance.now();
    const latencyMs = parseFloat((tEnd - tStart).toFixed(2));

    // Output probabilities: Tensor(shape=[1, 2])
    const probs = output.probabilities.data;
    const pd = probs[1]; // Probabilidad clase 1 (Default)

    const threshold = this.meta.threshold || 0.18;
    const decision = pd >= threshold ? "RECHAZAR" : "APROBAR";
    const band = this.riskBand(pd);
    const factors = this.computeExplainFactors(form, pd, inst);

    return {
      probability: pd,
      decision: decision,
      risk_band: band,
      threshold: threshold,
      installment: inst,
      factors: factors,
      latency_ms: latencyMs
    };
  }
}

// Exportar global para el navegador
window.OnnxRiskEngine = OnnxRiskEngine;
