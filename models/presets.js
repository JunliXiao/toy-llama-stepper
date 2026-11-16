/**
 * Toy Llama Stepper - 預設模型定義 (Pure Model Weights & Architecture)
 * 模型名稱與描述已抽離至 js/i18n.js，支援無伺服器直接開啟 (file://) 與 GitHub Pages (https://)
 */
(function (global) {
  'use strict';

  // 基本模型超參數（保持與專案預設一致）
  const defaultCfg = {
    d: 16,
    L: 4,
    Hq: 4,
    Hkv: 2,
    hd: 4,
    ffm: 3,
    theta: 10000
  };

  // PRNG 與亂數產生工具（保證可重現）
  const rng = s => () => {
    s |= 0;
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = r => (r() + r() + r() + r() - 2) * 1.732;
  const rmat = (r, n, m, s) => Array.from({ length: n }, () => Array.from({ length: m }, () => gauss(r) * s));
  const ones = n => Array(n).fill(1);
  const rms = (X, g) => X.map(r => {
    const s = Math.sqrt(r.reduce((a, v) => a + v * v, 0) / r.length + 1e-6);
    return r.map((v, i) => v / s * g[i]);
  });

  // 矩陣轉置與乘法工具
  const tr = A => A[0].map((_, j) => A.map(r => r[j]));
  const matMul = (A, B) => A.map(row => tr(B).map(col => row.reduce((sum, v, k) => sum + v * col[k], 0)));

  // 高斯消去法求反矩陣 (n x n)
  function matInv(M, reg = 1e-4) {
    const n = M.length;
    const A = M.map((r, i) => r.map((v, j) => v + (i === j ? reg : 0)));
    const I = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
    for (let i = 0; i < n; i++) {
      let pivot = i;
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(A[r][i]) > Math.abs(A[pivot][i])) pivot = r;
      }
      if (Math.abs(A[pivot][i]) < 1e-12) continue;
      [A[i], A[pivot]] = [A[pivot], A[i]];
      [I[i], I[pivot]] = [I[pivot], I[i]];
      const div = A[i][i];
      for (let j = 0; j < n; j++) { A[i][j] /= div; I[i][j] /= div; }
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const factor = A[r][i];
        for (let j = 0; j < n; j++) {
          A[r][j] -= factor * A[i][j];
          I[r][j] -= factor * I[i][j];
        }
      }
    }
    return I;
  }

  /**
   * 建立基準模型 (Baseline)：以指定 seed 生成隨機權重
   */
  function createBaselineModel(seed = 7) {
    const { d, L, Hq, Hkv, hd, ffm } = defaultCfg;
    const ff = d * ffm;
    const r = rng(seed);

    const layers = [];
    for (let l = 0; l < L; l++) {
      layers.push({
        g1: ones(d),
        g2: ones(d),
        Wq: rmat(r, d, Hq * hd, 1 / Math.sqrt(d)),
        Wk: rmat(r, d, Hkv * hd, 1 / Math.sqrt(d)),
        Wv: rmat(r, d, Hkv * hd, 1 / Math.sqrt(d)),
        Wo: rmat(r, Hq * hd, d, 1 / Math.sqrt(Hq * hd)),
        Wg: rmat(r, d, ff, 1 / Math.sqrt(d)),
        Wu: rmat(r, d, ff, 1 / Math.sqrt(d)),
        Wd: rmat(r, ff, d, 1 / Math.sqrt(ff))
      });
    }

    return {
      id: 'baseline',
      config: { ...defaultCfg },
      gf: ones(d),
      layers
    };
  }

  /**
   * 建立機制模型 (Previous-Token Head)：
   * 基於 Baseline，將 Layer 0 的 Head 0 特異化為「前一個 Token 注意力頭」
   */
  function createPrevTokenModel(baseline) {
    const model = JSON.parse(JSON.stringify(baseline));
    model.id = 'prev_token';

    const { d } = defaultCfg;
    const l0 = model.layers[0];

    const T_ref = 8;
    const xRng = rng(7 + 999);
    const X_ref = rmat(xRng, T_ref, d, 1);
    const N_ref = rms(X_ref, l0.g1);

    const mag = 3.6;
    const Q_target = Array.from({ length: T_ref }, () => [mag, 0, 0, 0]);
    const K_target = Array.from({ length: T_ref }, () => [mag * Math.cos(1), 0, mag * Math.sin(1), 0]);

    const NNt = matMul(N_ref, tr(N_ref));
    const invNNt = matInv(NNt, 1e-3);
    const pinvN = matMul(tr(N_ref), invNNt);

    const Wq_head0 = matMul(pinvN, Q_target);
    const Wk_head0 = matMul(pinvN, K_target);

    for (let i = 0; i < d; i++) {
      for (let j = 0; j < 4; j++) {
        l0.Wq[i][j] = Wq_head0[i][j];
        l0.Wk[i][j] = Wk_head0[i][j];
      }
    }

    return model;
  }

  /**
   * 建立機制模型 (SwiGLU 特徵開關)：
   * 基於 Baseline，將 Layer 0 MLP 前饋神經元設計為特徵檢測器與 Key-Value 記憶儲存
   */
  function createSwiGLUGateModel(baseline) {
    const model = JSON.parse(JSON.stringify(baseline));
    model.id = 'swiglu_gate';

    const { d } = defaultCfg;
    const l0 = model.layers[0];

    for (let k = 0; k < d; k++) {
      l0.Wg[k][0] = k === 0 ? 4.0 : 0.0;
      l0.Wu[k][0] = k === 0 ? 2.0 : 0.0;
      l0.Wd[0][k] = k === 0 ? 1.0 : 0.0;
    }

    for (let k = 0; k < d; k++) {
      l0.Wg[k][1] = k === 0 ? -4.0 : 0.0;
      l0.Wu[k][1] = k === 0 ? -2.0 : 0.0;
      l0.Wd[1][k] = k === 0 ? -1.0 : 0.0;
    }

    return model;
  }

  const baseline = createBaselineModel(7);
  const prevToken = createPrevTokenModel(baseline);
  const swigluGate = createSwiGLUGateModel(baseline);

  global.TOY_MODELS = {
    baseline,
    prev_token: prevToken,
    swiglu_gate: swigluGate
  };

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);