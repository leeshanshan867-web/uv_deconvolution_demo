import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, Play, Beaker, Activity, FileText, Download, Info, ChevronRight, Sparkles } from 'lucide-react';

// ============================================================
// MATH HELPERS — Linear algebra for spectral deconvolution
// ============================================================

// Matrix multiply: A (m×n) * B (n×p) -> (m×p)
function matMul(A, B) {
  const m = A.length, n = A[0].length, p = B[0].length;
  const C = Array.from({ length: m }, () => new Array(p).fill(0));
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < p; j++) {
        C[i][j] += aik * B[k][j];
      }
    }
  }
  return C;
}

function transpose(A) {
  const m = A.length, n = A[0].length;
  const T = Array.from({ length: n }, () => new Array(m));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      T[j][i] = A[i][j];
  return T;
}

// Solve A x = b via Gauss-Jordan with partial pivoting (A is square)
function solveLinear(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++)
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    [M[i], M[maxRow]] = [M[maxRow], M[i]];
    if (Math.abs(M[i][i]) < 1e-12) return null;
    for (let k = i + 1; k < n; k++) {
      const f = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// Non-negative least squares (simplified active-set / projected gradient)
// Solves: min ||A x - b||^2 s.t. x >= 0
function nnls(A, b, maxIter = 200) {
  const n = A[0].length;
  const At = transpose(A);
  const AtA = matMul(At, A);
  const Atb = matMul(At, b.map(v => [v])).map(r => r[0]);

  let x = new Array(n).fill(0);
  // Projected gradient descent with adaptive step
  let stepSize = 1.0 / (Math.max(...AtA.map((r, i) => r[i])) + 1e-9);
  for (let iter = 0; iter < maxIter; iter++) {
    const grad = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = -Atb[i];
      for (let j = 0; j < n; j++) s += AtA[i][j] * x[j];
      grad[i] = s;
    }
    let maxChange = 0;
    for (let i = 0; i < n; i++) {
      const newVal = Math.max(0, x[i] - stepSize * grad[i]);
      maxChange = Math.max(maxChange, Math.abs(newVal - x[i]));
      x[i] = newVal;
    }
    if (maxChange < 1e-8) break;
  }
  return x;
}

// CLS: Given pure spectra S (n_wavelengths × n_species) and mixture spectra D (n_wavelengths × n_times),
// solve C = (S^T S)^-1 S^T D for each time point. Returns C (n_species × n_times).
function classicalLeastSquares(S, D, nonNeg = true) {
  const nT = D[0].length;
  const nSpecies = S[0].length;
  const C = Array.from({ length: nSpecies }, () => new Array(nT).fill(0));
  for (let t = 0; t < nT; t++) {
    const dCol = D.map(row => row[t]);
    const c = nonNeg ? nnls(S, dCol) : (() => {
      const StS = matMul(transpose(S), S);
      const Std = matMul(transpose(S), dCol.map(v => [v])).map(r => r[0]);
      return solveLinear(StS, Std);
    })();
    for (let i = 0; i < nSpecies; i++) C[i][t] = c[i];
  }
  return C;
}

// Simple MCR-ALS: given mixture matrix D (n_wavelengths × n_times), n_components,
// alternately solve for C and S with non-negativity. Initialize S with NMF-style random or PCA-derived.
function mcrAls(D, nComp, maxIter = 80, initS = null) {
  const nW = D.length;
  const nT = D[0].length;
  // Initialize spectra S (nW × nComp) — use evenly spaced columns of D as starting guess
  let S;
  if (initS) {
    S = initS.map(row => [...row]);
  } else {
    S = Array.from({ length: nW }, () => new Array(nComp).fill(0));
    for (let k = 0; k < nComp; k++) {
      const tIdx = Math.floor((k + 0.5) * nT / nComp);
      for (let i = 0; i < nW; i++) S[i][k] = Math.max(0, D[i][tIdx]) + 0.01 * Math.random();
    }
  }

  let C = Array.from({ length: nComp }, () => new Array(nT).fill(0));
  let prevErr = Infinity;

  for (let iter = 0; iter < maxIter; iter++) {
    // Update C: for each time t, solve S c = D[:,t]
    for (let t = 0; t < nT; t++) {
      const dCol = D.map(row => row[t]);
      const c = nnls(S, dCol, 100);
      for (let k = 0; k < nComp; k++) C[k][t] = c[k];
    }
    // Update S: for each wavelength w, solve C^T s = D[w,:]
    const Ct = transpose(C); // nT × nComp
    for (let w = 0; w < nW; w++) {
      const dRow = D[w];
      const s = nnls(Ct, dRow, 100);
      for (let k = 0; k < nComp; k++) S[w][k] = s[k];
    }
    // Normalize spectra (unit max) and rescale C to absorb scale — improves stability
    for (let k = 0; k < nComp; k++) {
      let maxS = 0;
      for (let w = 0; w < nW; w++) maxS = Math.max(maxS, S[w][k]);
      if (maxS > 1e-10) {
        for (let w = 0; w < nW; w++) S[w][k] /= maxS;
        for (let t = 0; t < nT; t++) C[k][t] *= maxS;
      }
    }
    // Compute reconstruction error
    const Drec = matMul(S, C);
    let err = 0;
    for (let i = 0; i < nW; i++)
      for (let j = 0; j < nT; j++)
        err += (D[i][j] - Drec[i][j]) ** 2;
    err = Math.sqrt(err / (nW * nT));
    if (Math.abs(prevErr - err) / (prevErr + 1e-12) < 1e-6) break;
    prevErr = err;
  }
  return { C, S, err: prevErr };
}

// ============================================================
// SYNTHETIC DATA: A → B → C → D consecutive reaction
// ============================================================
function generateSyntheticData() {
  const wavelengths = [];
  for (let w = 250; w <= 500; w += 5) wavelengths.push(w);
  const times = [];
  for (let t = 0; t <= 60; t += 2) times.push(t);

  // Gaussian peaks for 4 species
  const gauss = (x, mu, sigma, amp) => amp * Math.exp(-((x - mu) ** 2) / (2 * sigma ** 2));
  const peaks = [
    { mu: 290, sigma: 18, amp: 1.0 },  // A
    { mu: 340, sigma: 22, amp: 0.85 }, // B
    { mu: 390, sigma: 20, amp: 1.1 },  // C
    { mu: 440, sigma: 25, amp: 0.7 },  // D
  ];
  const pureSpectra = wavelengths.map(w =>
    peaks.map(p => gauss(w, p.mu, p.sigma, p.amp))
  );

  // Kinetics: A -> B -> C -> D, all first-order
  const k1 = 0.08, k2 = 0.05, k3 = 0.04;
  const A0 = 1.0;
  const concentrations = times.map(t => {
    const A = A0 * Math.exp(-k1 * t);
    // Bateman equations for sequential first-order
    const B = (A0 * k1 / (k2 - k1)) * (Math.exp(-k1 * t) - Math.exp(-k2 * t));
    const C = A0 * k1 * k2 * (
      Math.exp(-k1 * t) / ((k2 - k1) * (k3 - k1)) +
      Math.exp(-k2 * t) / ((k1 - k2) * (k3 - k2)) +
      Math.exp(-k3 * t) / ((k1 - k3) * (k2 - k3))
    );
    const D = A0 - A - B - C;
    return [A, Math.max(0, B), Math.max(0, C), Math.max(0, D)];
  });

  // Mixture spectra at each time (with small noise)
  const mixtures = wavelengths.map((w, wi) =>
    times.map((t, ti) => {
      let v = 0;
      for (let k = 0; k < 4; k++) v += pureSpectra[wi][k] * concentrations[ti][k];
      return v + (Math.random() - 0.5) * 0.01;
    })
  );

  return { wavelengths, times, pureSpectra, concentrations, mixtures };
}

// ============================================================
// CSV PARSING
// ============================================================
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  const rows = lines.map(l => l.split(/[,\t;]/).map(c => c.trim()));
  return rows;
}

// Expected mixture format: first row = times (first cell ignored / "wavelength"),
// first column = wavelengths, rest = absorbance.
function parseMixtureCSV(text) {
  const rows = parseCSV(text);
  const header = rows[0];
  const times = header.slice(1).map(Number);
  const wavelengths = [];
  const data = [];
  for (let i = 1; i < rows.length; i++) {
    wavelengths.push(Number(rows[i][0]));
    data.push(rows[i].slice(1).map(Number));
  }
  return { wavelengths, times, mixtures: data };
}

// Expected reference format: first row header (wavelength, sp1, sp2, ...),
// first column = wavelengths, columns = pure spectra.
function parseReferenceCSV(text) {
  const rows = parseCSV(text);
  const header = rows[0];
  const speciesNames = header.slice(1);
  const wavelengths = [];
  const spectra = [];
  for (let i = 1; i < rows.length; i++) {
    wavelengths.push(Number(rows[i][0]));
    spectra.push(rows[i].slice(1).map(Number));
  }
  return { wavelengths, speciesNames, spectra };
}

// ============================================================
// MAIN COMPONENT
// ============================================================
const COLORS = ['#c2410c', '#0e7490', '#a16207', '#7c2d12', '#365314', '#831843'];
const SPECIES_LABELS = ['A', 'B', 'C', 'D', 'E', 'F'];

export default function App() {
  const [mode, setMode] = useState('demo'); // 'demo' | 'upload'
  const [hasReferences, setHasReferences] = useState(true);
  const [nComponents, setNComponents] = useState(4);
  const [mixtureData, setMixtureData] = useState(null);
  const [referenceData, setReferenceData] = useState(null);
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTimeIdx, setSelectedTimeIdx] = useState(0);

  // Auto-load demo data on mount
  useEffect(() => {
    const synth = generateSyntheticData();
    setMixtureData({
      wavelengths: synth.wavelengths,
      times: synth.times,
      mixtures: synth.mixtures,
    });
    setReferenceData({
      wavelengths: synth.wavelengths,
      speciesNames: ['A', 'B', 'C', 'D'],
      spectra: synth.pureSpectra,
      trueConcentrations: synth.concentrations,
    });
  }, []);

  const handleMixtureUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = parseMixtureCSV(ev.target.result);
        setMixtureData(parsed);
        setMode('upload');
        setError(null);
        setResults(null);
      } catch (err) {
        setError('Could not parse mixture CSV. Expected: header row of times, first column of wavelengths.');
      }
    };
    reader.readAsText(file);
  };

  const handleReferenceUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = parseReferenceCSV(ev.target.result);
        setReferenceData(parsed);
        setHasReferences(true);
        setError(null);
        setResults(null);
      } catch (err) {
        setError('Could not parse reference CSV.');
      }
    };
    reader.readAsText(file);
  };

  const runDeconvolution = async () => {
    if (!mixtureData) return;
    setRunning(true);
    setError(null);
    await new Promise(r => setTimeout(r, 50)); // let UI update

    try {
      const D = mixtureData.mixtures;
      let C, S, method, speciesNames;

      if (hasReferences && referenceData && referenceData.spectra) {
        // CLS path
        S = referenceData.spectra;
        C = classicalLeastSquares(S, D, true);
        method = 'Classical Least Squares (Beer-Lambert, NNLS)';
        speciesNames = referenceData.speciesNames;
      } else {
        // MCR-ALS path
        const out = mcrAls(D, nComponents, 80);
        C = out.C;
        S = out.S;
        method = `MCR-ALS (${nComponents} components, RMSE: ${out.err.toFixed(4)})`;
        speciesNames = Array.from({ length: nComponents }, (_, i) => SPECIES_LABELS[i]);
      }

      setResults({ C, S, method, speciesNames });
    } catch (err) {
      setError('Deconvolution failed: ' + err.message);
    }
    setRunning(false);
  };

  // Build chart data
  const concentrationChartData = useMemo(() => {
    if (!results || !mixtureData) return [];
    return mixtureData.times.map((t, ti) => {
      const row = { time: t };
      results.speciesNames.forEach((name, i) => {
        row[name] = results.C[i][ti];
      });
      // include true values if demo
      if (mode === 'demo' && referenceData?.trueConcentrations) {
        results.speciesNames.forEach((name, i) => {
          row[name + '_true'] = referenceData.trueConcentrations[ti][i];
        });
      }
      return row;
    });
  }, [results, mixtureData, mode, referenceData]);

  const spectraChartData = useMemo(() => {
    if (!results || !mixtureData) return [];
    return mixtureData.wavelengths.map((w, wi) => {
      const row = { wavelength: w };
      results.speciesNames.forEach((name, i) => {
        row[name] = results.S[wi][i];
      });
      return row;
    });
  }, [results, mixtureData]);

  const mixtureChartData = useMemo(() => {
    if (!mixtureData) return [];
    return mixtureData.wavelengths.map((w, wi) => {
      const row = { wavelength: w };
      mixtureData.times.forEach((t, ti) => {
        row['t' + ti] = mixtureData.mixtures[wi][ti];
      });
      return row;
    });
  }, [mixtureData]);

  const downloadResults = () => {
    if (!results || !mixtureData) return;
    let csv = 'time,' + results.speciesNames.join(',') + '\n';
    mixtureData.times.forEach((t, ti) => {
      csv += t + ',' + results.speciesNames.map((_, i) => results.C[i][ti].toFixed(6)).join(',') + '\n';
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'concentrations.csv';
    a.click();
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f5f1e8',
      fontFamily: 'Georgia, "Times New Roman", serif',
      color: '#2a1810',
      padding: '2rem',
    }}>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .fade-in { animation: fadeIn 0.5s ease-out; }
        .pulse { animation: pulse 1.5s ease-in-out infinite; }
        .btn-primary {
          background: #2a1810; color: #f5f1e8; border: none; padding: 0.75rem 1.5rem;
          font-family: inherit; font-size: 0.95rem; cursor: pointer; letter-spacing: 0.05em;
          text-transform: uppercase; transition: all 0.2s; display: inline-flex; align-items: center; gap: 0.5rem;
        }
        .btn-primary:hover:not(:disabled) { background: #c2410c; transform: translateY(-1px); }
        .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-ghost {
          background: transparent; color: #2a1810; border: 1px solid #2a1810; padding: 0.5rem 1rem;
          font-family: inherit; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 0.5rem;
        }
        .btn-ghost:hover { background: #2a1810; color: #f5f1e8; }
        .toggle-btn {
          padding: 0.5rem 1rem; border: 1px solid #2a1810; background: transparent; cursor: pointer;
          font-family: inherit; transition: all 0.15s;
        }
        .toggle-btn.active { background: #2a1810; color: #f5f1e8; }
        .file-input { display: none; }
        .file-label {
          display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1rem;
          border: 1px dashed #2a1810; cursor: pointer; transition: all 0.2s; font-size: 0.9rem;
        }
        .file-label:hover { background: #2a1810; color: #f5f1e8; }
        .panel {
          background: #faf6ed; border: 1px solid #d4c5a9; padding: 1.5rem; margin-bottom: 1.5rem;
        }
        .section-num {
          font-size: 0.75rem; letter-spacing: 0.2em; color: #8b6f47; text-transform: uppercase;
          margin-bottom: 0.5rem; display: block;
        }
        .h-display { font-size: 2rem; line-height: 1.1; margin: 0 0 0.5rem 0; font-weight: 400; letter-spacing: -0.01em; }
        .stat-card {
          background: #faf6ed; border-left: 3px solid #c2410c; padding: 1rem 1.25rem;
        }
        .stat-num { font-size: 1.75rem; font-family: 'Courier New', monospace; color: #2a1810; }
        .stat-label { font-size: 0.7rem; letter-spacing: 0.15em; color: #8b6f47; text-transform: uppercase; }
        .ornamental-rule {
          border: none; border-top: 1px solid #d4c5a9; margin: 2rem 0; position: relative;
        }
        .ornamental-rule::after {
          content: '◆'; position: absolute; top: -0.7em; left: 50%; transform: translateX(-50%);
          background: #f5f1e8; padding: 0 0.75rem; color: #c2410c; font-size: 0.8rem;
        }
      `}</style>

      <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
        {/* HEADER */}
        <header style={{ borderBottom: '2px solid #2a1810', paddingBottom: '1.5rem', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <div>
              <div style={{ fontSize: '0.7rem', letterSpacing: '0.3em', color: '#8b6f47', marginBottom: '0.25rem' }}>
                SPECTRAL ANALYSIS · VOL. I
              </div>
              <h1 style={{ fontSize: '3rem', lineHeight: '1', margin: 0, letterSpacing: '-0.02em', fontWeight: 400 }}>
                UV-Vis <em style={{ color: '#c2410c' }}>Deconvolution</em>
              </h1>
              <div style={{ marginTop: '0.5rem', fontSize: '0.95rem', color: '#5c4a3a', fontStyle: 'italic' }}>
                Resolving concentration profiles of overlapping species through time
              </div>
            </div>
            <div style={{ textAlign: 'right', fontSize: '0.8rem', color: '#8b6f47', fontFamily: 'Courier New, monospace' }}>
              <div>BEER—LAMBERT</div>
              <div>MCR—ALS · CLS · NNLS</div>
            </div>
          </div>
        </header>

        {/* INTRO */}
        <div className="panel fade-in" style={{ background: '#2a1810', color: '#f5f1e8', borderColor: '#2a1810' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
            <div>
              <span className="section-num" style={{ color: '#d4a574' }}>§ Method</span>
              <h2 style={{ margin: '0 0 0.75rem 0', fontWeight: 400, fontSize: '1.4rem' }}>
                Two paths, one goal
              </h2>
              <p style={{ margin: 0, lineHeight: 1.6, fontSize: '0.95rem' }}>
                Given a series of mixture spectra <code style={{ background: '#3a2820', padding: '1px 6px' }}>D(λ, t)</code> from
                a reaction, recover concentration profiles <code style={{ background: '#3a2820', padding: '1px 6px' }}>C(t)</code>
                {' '}for each species. The route depends on what you already know.
              </p>
            </div>
            <div style={{ borderLeft: '1px solid #5c4a3a', paddingLeft: '1.5rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <strong style={{ color: '#d4a574' }}>If you have pure reference spectra →</strong> Classical Least Squares.
                Linear, fast, well-defined. One matrix solve per timepoint.
              </div>
              <div>
                <strong style={{ color: '#d4a574' }}>If you only have the mixtures →</strong> MCR-ALS. Iteratively
                discovers both pure spectra <em>and</em> concentrations using non-negativity constraints.
              </div>
            </div>
          </div>
        </div>

        {/* CONFIGURATION PANEL */}
        <div className="panel fade-in">
          <span className="section-num">§ I — Configuration</span>
          <h2 className="h-display">Set up your analysis</h2>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '1.5rem' }}>
            {/* Data source */}
            <div>
              <label style={{ fontSize: '0.85rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#5c4a3a', display: 'block', marginBottom: '0.75rem' }}>
                Data source
              </label>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <button
                  className={`toggle-btn ${mode === 'demo' ? 'active' : ''}`}
                  onClick={() => {
                    setMode('demo');
                    const synth = generateSyntheticData();
                    setMixtureData({
                      wavelengths: synth.wavelengths,
                      times: synth.times,
                      mixtures: synth.mixtures,
                    });
                    setReferenceData({
                      wavelengths: synth.wavelengths,
                      speciesNames: ['A', 'B', 'C', 'D'],
                      spectra: synth.pureSpectra,
                      trueConcentrations: synth.concentrations,
                    });
                    setResults(null);
                  }}
                >
                  <Sparkles size={14} style={{ display: 'inline', marginRight: 4 }} />
                  Synthetic demo
                </button>
                <label className="file-label">
                  <Upload size={14} />
                  Upload mixture CSV
                  <input type="file" accept=".csv,.txt,.tsv" className="file-input" onChange={handleMixtureUpload} />
                </label>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#8b6f47', fontStyle: 'italic' }}>
                {mode === 'demo'
                  ? 'Loaded: simulated A→B→C→D consecutive first-order reaction, 51 wavelengths × 31 timepoints, with noise.'
                  : mixtureData
                    ? `Loaded: ${mixtureData.wavelengths.length} wavelengths × ${mixtureData.times.length} timepoints`
                    : 'CSV format: header row of times, first column of wavelengths.'}
              </div>
            </div>

            {/* Method */}
            <div>
              <label style={{ fontSize: '0.85rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#5c4a3a', display: 'block', marginBottom: '0.75rem' }}>
                Reference spectra
              </label>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                <button
                  className={`toggle-btn ${hasReferences ? 'active' : ''}`}
                  onClick={() => setHasReferences(true)}
                >
                  Yes — use CLS
                </button>
                <button
                  className={`toggle-btn ${!hasReferences ? 'active' : ''}`}
                  onClick={() => setHasReferences(false)}
                >
                  No — use MCR-ALS
                </button>
              </div>
              {hasReferences ? (
                <div>
                  <label className="file-label" style={{ marginBottom: '0.5rem' }}>
                    <FileText size={14} />
                    Upload reference spectra
                    <input type="file" accept=".csv,.txt,.tsv" className="file-input" onChange={handleReferenceUpload} />
                  </label>
                  <div style={{ fontSize: '0.8rem', color: '#8b6f47', fontStyle: 'italic' }}>
                    {mode === 'demo' && referenceData
                      ? `Using demo references: ${referenceData.speciesNames.join(', ')}`
                      : 'CSV: header row "wavelength, species1, species2, ...", same wavelength grid as mixture.'}
                  </div>
                </div>
              ) : (
                <div>
                  <label style={{ fontSize: '0.85rem', color: '#5c4a3a', display: 'block', marginBottom: '0.5rem' }}>
                    Number of components: <strong style={{ color: '#c2410c' }}>{nComponents}</strong>
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="6"
                    value={nComponents}
                    onChange={e => setNComponents(Number(e.target.value))}
                    style={{ width: '100%', accentColor: '#c2410c' }}
                  />
                  <div style={{ fontSize: '0.8rem', color: '#8b6f47', fontStyle: 'italic', marginTop: '0.5rem' }}>
                    MCR-ALS will resolve this many spectra + concentration profiles from the data alone.
                  </div>
                </div>
              )}
            </div>
          </div>

          <hr className="ornamental-rule" />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: '#5c4a3a' }}>
              {hasReferences
                ? <>Using <strong>CLS</strong> with non-negativity (NNLS). Direct, deterministic.</>
                : <>Using <strong>MCR-ALS</strong>. Iterative; results depend on initialization and number of components.</>}
            </div>
            <button className="btn-primary" onClick={runDeconvolution} disabled={running || !mixtureData}>
              {running ? <><span className="pulse">●</span> Solving</> : <><Play size={14} /> Run deconvolution</>}
            </button>
          </div>

          {error && (
            <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: '#fef2f2', border: '1px solid #c2410c', color: '#7c2d12' }}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* RAW MIXTURE PREVIEW */}
        {mixtureData && (
          <div className="panel fade-in">
            <span className="section-num">§ II — Observation</span>
            <h2 className="h-display">The measured mixture <em style={{ color: '#c2410c' }}>D(λ, t)</em></h2>
            <p style={{ color: '#5c4a3a', marginTop: 0 }}>
              {mixtureData.times.length} spectra, one per timepoint. Each curve shows total absorbance — a sum of contributions from all species.
            </p>
            <div style={{ height: 300, marginTop: '1rem' }}>
              <ResponsiveContainer>
                <LineChart data={mixtureChartData}>
                  <CartesianGrid stroke="#d4c5a9" strokeDasharray="2 4" />
                  <XAxis dataKey="wavelength" stroke="#5c4a3a" label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5, fill: '#5c4a3a' }} />
                  <YAxis stroke="#5c4a3a" label={{ value: 'Absorbance', angle: -90, position: 'insideLeft', fill: '#5c4a3a' }} />
                  {mixtureData.times.map((t, ti) => (
                    <Line
                      key={ti}
                      type="monotone"
                      dataKey={'t' + ti}
                      stroke={`hsl(${20 + ti * 2}, 70%, ${50 - ti * 0.5}%)`}
                      strokeWidth={1}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{ fontSize: '0.75rem', color: '#8b6f47', textAlign: 'center', fontStyle: 'italic' }}>
              Color gradient: dark = early time, light = late time
            </div>
          </div>
        )}

        {/* RESULTS */}
        {results && (
          <>
            <div className="panel fade-in">
              <span className="section-num">§ III — Resolution</span>
              <h2 className="h-display">Concentration profiles <em style={{ color: '#c2410c' }}>C(t)</em></h2>
              <p style={{ color: '#5c4a3a', marginTop: 0, marginBottom: '0.5rem' }}>
                Method: <strong>{results.method}</strong>
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${results.speciesNames.length}, 1fr)`, gap: '0.75rem', margin: '1.5rem 0' }}>
                {results.speciesNames.map((name, i) => {
                  const cMax = Math.max(...results.C[i]);
                  const cFinal = results.C[i][results.C[i].length - 1];
                  return (
                    <div key={i} className="stat-card" style={{ borderLeftColor: COLORS[i % COLORS.length] }}>
                      <div className="stat-label">Species {name}</div>
                      <div className="stat-num">{cMax.toFixed(3)}</div>
                      <div style={{ fontSize: '0.75rem', color: '#8b6f47' }}>peak · final {cFinal.toFixed(3)}</div>
                    </div>
                  );
                })}
              </div>

              <div style={{ height: 380 }}>
                <ResponsiveContainer>
                  <LineChart data={concentrationChartData}>
                    <CartesianGrid stroke="#d4c5a9" strokeDasharray="2 4" />
                    <XAxis dataKey="time" stroke="#5c4a3a" label={{ value: 'Time', position: 'insideBottom', offset: -5, fill: '#5c4a3a' }} />
                    <YAxis stroke="#5c4a3a" label={{ value: 'Concentration', angle: -90, position: 'insideLeft', fill: '#5c4a3a' }} />
                    <Tooltip contentStyle={{ background: '#faf6ed', border: '1px solid #2a1810', fontFamily: 'Georgia, serif' }} />
                    <Legend />
                    {results.speciesNames.map((name, i) => (
                      <Line
                        key={name}
                        type="monotone"
                        dataKey={name}
                        stroke={COLORS[i % COLORS.length]}
                        strokeWidth={2.5}
                        dot={{ r: 2.5, fill: COLORS[i % COLORS.length] }}
                        isAnimationActive={true}
                      />
                    ))}
                    {mode === 'demo' && referenceData?.trueConcentrations && results.speciesNames.map((name, i) => (
                      <Line
                        key={name + '_true'}
                        type="monotone"
                        dataKey={name + '_true'}
                        stroke={COLORS[i % COLORS.length]}
                        strokeWidth={1}
                        strokeDasharray="4 4"
                        dot={false}
                        isAnimationActive={false}
                        legendType="none"
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {mode === 'demo' && (
                <div style={{ fontSize: '0.8rem', color: '#8b6f47', textAlign: 'center', fontStyle: 'italic', marginTop: '0.5rem' }}>
                  Solid lines: recovered concentrations. Dashed lines: ground truth (demo only).
                </div>
              )}

              <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.75rem' }}>
                <button className="btn-ghost" onClick={downloadResults}>
                  <Download size={14} /> Export concentrations.csv
                </button>
              </div>
            </div>

            {/* RESOLVED SPECTRA */}
            <div className="panel fade-in">
              <span className="section-num">§ IV — Pure spectra</span>
              <h2 className="h-display">Resolved species spectra <em style={{ color: '#c2410c' }}>S(λ)</em></h2>
              <p style={{ color: '#5c4a3a', marginTop: 0 }}>
                {hasReferences
                  ? 'These are the reference spectra you provided, unchanged.'
                  : 'MCR-ALS extracted these pure-component spectra directly from the time-resolved mixture.'}
              </p>
              <div style={{ height: 320, marginTop: '1rem' }}>
                <ResponsiveContainer>
                  <LineChart data={spectraChartData}>
                    <CartesianGrid stroke="#d4c5a9" strokeDasharray="2 4" />
                    <XAxis dataKey="wavelength" stroke="#5c4a3a" label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5, fill: '#5c4a3a' }} />
                    <YAxis stroke="#5c4a3a" label={{ value: 'ε (or norm.)', angle: -90, position: 'insideLeft', fill: '#5c4a3a' }} />
                    <Tooltip contentStyle={{ background: '#faf6ed', border: '1px solid #2a1810' }} />
                    <Legend />
                    {results.speciesNames.map((name, i) => (
                      <Line
                        key={name}
                        type="monotone"
                        dataKey={name}
                        stroke={COLORS[i % COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}

        {/* SOFTWARE GUIDE */}
        <div className="panel fade-in">
          <span className="section-num">§ V — Real-world software</span>
          <h2 className="h-display">When you outgrow this app</h2>
          <p style={{ color: '#5c4a3a' }}>
            This in-browser tool implements the core algorithms — but for production use, complex constraints, or large datasets,
            consider these established packages.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
            <div style={{ padding: '1rem', background: '#faf6ed', borderLeft: '3px solid #c2410c' }}>
              <div style={{ fontSize: '0.75rem', letterSpacing: '0.15em', color: '#8b6f47', textTransform: 'uppercase' }}>Python · Free</div>
              <h3 style={{ margin: '0.25rem 0 0.5rem 0', fontWeight: 500 }}>pyMCR (NIST)</h3>
              <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.5 }}>
                The reference open-source implementation. Native sklearn integration, custom regressors, flexible constraints.
                Ideal if you want reproducible scripts.
              </p>
              <code style={{ display: 'block', marginTop: '0.5rem', fontSize: '0.8rem', color: '#7c2d12' }}>pip install pyMCR</code>
            </div>
            <div style={{ padding: '1rem', background: '#faf6ed', borderLeft: '3px solid #0e7490' }}>
              <div style={{ fontSize: '0.75rem', letterSpacing: '0.15em', color: '#8b6f47', textTransform: 'uppercase' }}>MATLAB · Free GUI</div>
              <h3 style={{ margin: '0.25rem 0 0.5rem 0', fontWeight: 500 }}>MCR-ALS GUI (Tauler)</h3>
              <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.5 }}>
                The chemometrics gold standard. Hard kinetic modeling (fit A→B→C directly), trilinearity, closure.
                Most cited tool in this domain.
              </p>
            </div>
            <div style={{ padding: '1rem', background: '#faf6ed', borderLeft: '3px solid #a16207' }}>
              <div style={{ fontSize: '0.75rem', letterSpacing: '0.15em', color: '#8b6f47', textTransform: 'uppercase' }}>Commercial</div>
              <h3 style={{ margin: '0.25rem 0 0.5rem 0', fontWeight: 500 }}>ReactLab Kinetics / Specfit</h3>
              <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.5 }}>
                Purpose-built for UV-Vis kinetics with mechanistic fitting. Handles equilibria, intermediates, fast reactions.
              </p>
            </div>
            <div style={{ padding: '1rem', background: '#faf6ed', borderLeft: '3px solid #7c2d12' }}>
              <div style={{ fontSize: '0.75rem', letterSpacing: '0.15em', color: '#8b6f47', textTransform: 'uppercase' }}>Origin · PeakFit</div>
              <h3 style={{ margin: '0.25rem 0 0.5rem 0', fontWeight: 500 }}>Curve fitting suites</h3>
              <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.5 }}>
                Best when bands are well-separated Gaussians/Lorentzians. Less suited to overlapping kinetic series.
              </p>
            </div>
          </div>
        </div>

        <footer style={{ textAlign: 'center', padding: '2rem 0', fontSize: '0.8rem', color: '#8b6f47', fontStyle: 'italic' }}>
          ◆ ◆ ◆
          <div style={{ marginTop: '0.5rem' }}>Built with Beer—Lambert's law and a fondness for old chemistry texts</div>
        </footer>
      </div>
    </div>
  );
}
