/**
 * Toy Llama Stepper - 預設模型定義
 * 支援無伺服器直接開啟 (file://) 與 GitHub Pages (https://)
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

  // PRNG 與亂數產生工具（與 index.html 演算法一致，保證可重現）
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
      name: '基準模型：Random Baseline, Seed 7',
      desc: '標準隨機初始化模型（Seed 7）。注意力與前饋特徵呈均勻高斯分佈，展示一般 Transformer 數值流動。',
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
    // 深拷貝模型結構
    const model = JSON.parse(JSON.stringify(baseline));
    model.id = 'prev_token';
    model.name = '機制模型：Previous-Token Head';
    model.desc = '將 Layer 0 的 Head 0 設計為專注於「前一個 Token (t-1)」的注意力頭。請切換至「Causal mask + Softmax」步驟並觀察 Head 0 的次對角線注意力高權重。';

    const { d } = defaultCfg;
    const l0 = model.layers[0];

    // 使用預設種子生成輸入 X0 的參考代表點 (T=8)
    const T_ref = 8;
    const xRng = rng(7 + 999);
    const X_ref = rmat(xRng, T_ref, d, 1);
    const N_ref = rms(X_ref, l0.g1); // RMSNorm 正規化後的參考輸入

    // 目標 Q 與 K 向量：利用 RoPE 旋轉頻率 θ_0 = 1 弧度的相位差
    // 設定 Q_0 落在 (dim 0, dim 2) 平面角度為 0，K_0 角度為 1 弧度
    // 經 RoPE 旋轉後：Q'(t) · K'(s) 正比於 cos((t - s) - 1)
    // 使得當 s = t - 1 (前一個 Token) 時，內積達到最大值 cos(0) = 1.0
    const mag = 3.6; // 控制注意力聚集強度 (經 Softmax 後 > 88%)
    const Q_target = Array.from({ length: T_ref }, () => [mag, 0, 0, 0]);
    const K_target = Array.from({ length: T_ref }, () => [mag * Math.cos(1), 0, mag * Math.sin(1), 0]);

    // 透過偽逆矩陣 N^+ = N^T (N N^T + λI)^-1 求解最佳 Wq 與 Wk 投影切片
    const NNt = matMul(N_ref, tr(N_ref));
    const invNNt = matInv(NNt, 1e-3);
    const pinvN = matMul(tr(N_ref), invNNt);

    const Wq_head0 = matMul(pinvN, Q_target); // 16 x 4
    const Wk_head0 = matMul(pinvN, K_target); // 16 x 4

    // 替換 Layer 0 Head 0 的 Wq (cols 0..3) 與 KV Group 0 的 Wk (cols 0..3)
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
   * 基於 Baseline，將 Layer 0 MLP 前饋神經元設計為特徵檢測器 (Feature Detector) 與 Key-Value 記憶儲存
   */
  function createSwiGLUGateModel(baseline) {
    const model = JSON.parse(JSON.stringify(baseline));
    model.id = 'swiglu_gate';
    model.name = '機制模型：SwiGLU 特徵開關';
    model.desc = '將 Layer 0 的 MLP 神經元特化為「特徵檢測器與記憶寫入」。第 0 號神經元專注檢測「維度 0 為正值」並飽和放行；第 1 號神經元檢測「維度 0 為負值」。請在「SwiGLU」與「down_proj」步驟觀察閘門 (SiLU) 的精準篩選。';

    const { d } = defaultCfg;
    const l0 = model.layers[0];

    // 特化 Neuron 0: 正特徵檢測器 (Positive Detector for Dim 0)
    // 當 x̂[0] > 0 時，Wg 投影值為大正數，SiLU(G) 飽和打開 (≈ G)；x̂[0] < 0 時關閉 (≈ 0)
    for (let k = 0; k < d; k++) {
      l0.Wg[k][0] = k === 0 ? 4.0 : 0.0;
      l0.Wu[k][0] = k === 0 ? 2.0 : 0.0;
      l0.Wd[0][k] = k === 0 ? 1.0 : 0.0;
    }

    // 特化 Neuron 1: 負特徵檢測器 (Negative Detector for Dim 0)
    // 當 x̂[0] < 0 時，Wg 投影值 (-4 * 負數) 為正數，SiLU(G) 飽和打開；x̂[0] > 0 時關閉
    for (let k = 0; k < d; k++) {
      l0.Wg[k][1] = k === 0 ? -4.0 : 0.0;
      l0.Wu[k][1] = k === 0 ? -2.0 : 0.0;
      l0.Wd[1][k] = k === 0 ? -1.0 : 0.0;
    }

    return model;
  }

  // 構建並註冊模型至全域 window.TOY_MODELS
  const baseline = createBaselineModel(7);
  const prevToken = createPrevTokenModel(baseline);
  const swigluGate = createSwiGLUGateModel(baseline);

  global.TOY_MODELS = {
    baseline,
    prev_token: prevToken,
    swiglu_gate: swigluGate
  };

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
