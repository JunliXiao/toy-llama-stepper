/**
 * Toy Llama Stepper - 純數學與推論計算模組 (Pure Math & Inference Forward Pass)
 */
(function (global) {
  'use strict';

  global.ToyLlama = global.ToyLlama || {};

  const rng = s => () => {
    s |= 0;
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = r => (r() + r() + r() + r() - 2) * 1.732;
  const rmat = (r, n, m, s) => Array.from({ length: n }, () => Array.from({ length: m }, () => gauss(r) * s));
  const tr = A => A[0].map((_, j) => A.map(r => r[j]));
  const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
  const nrm = v => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
  const rms = (X, g) => X.map(r => {
    const s = Math.sqrt(r.reduce((a, v) => a + v * v, 0) / r.length + 1e-6);
    return r.map((v, i) => (v / s) * g[i]);
  });
  const f2 = v => (isFinite(v) ? v.toFixed(2) : v < 0 ? '−∞' : '∞');
  const silu = z => z / (1 + Math.exp(-z));
  const ones = n => Array(n).fill(1);

  function rope(M, hd, theta) {
    const h = hd / 2;
    return M.map((r, t) => {
      const o = r.slice();
      for (let i = 0; i < h; i++) {
        const a = t * Math.pow(theta, (-2 * i) / hd),
          c = Math.cos(a),
          s = Math.sin(a);
        o[i] = r[i] * c - r[i + h] * s;
        o[i + h] = r[i] * s + r[i + h] * c;
      }
      return o;
    });
  }

  const item = (A, Bt, an, bn, div = 1) => {
    const m = A.map(a => Bt.map(b => a.reduce((s, v, k) => s + v * b[k], 0) / div));
    return {
      m,
      micro: (i, j) =>
        `${an}[t${i}] · ${bn}[${j}] = (` +
        A[i].map((v, k) => `${f2(v)}×${f2(Bt[j][k])}`).join(' + ') +
        `)${div != 1 ? ' / √d_h(' + f2(div) + ')' : ''} = <b>${f2(m[i][j])}</b>`
    };
  };

  function forward(customState) {
    const S = customState || global.ToyLlama.state;
    const { cfg, W, X0 } = S;
    const { T, d, Hq, Hkv, hd, L, ffm, theta } = cfg;
    const grp = Hq / Hkv;
    const stepsMeta = global.ToyLlama.steps;
    const stepsTpl = stepsMeta ? stepsMeta.templates : null;
    const statsFn = stepsMeta ? stepsMeta.stats : null;

    S.steps = [];
    S.ck = [];
    let x = X0;

    const P = (layer, name, tex, desc, tensors, st) =>
      S.steps.push({ layer, name, tex, desc, tensors, st, ci: S.ck.length - 1 });
    const R = n => [...Array(n).keys()];

    // 1. Input X0
    S.ck.push({ x, label: 'X₀' });
    const tX0 = stepsTpl ? stepsTpl.input_x0 : { name: 'Input X₀', tex: '', desc: '' };
    P(-1, tX0.name, tX0.tex, tX0.desc, [{ label: 'X₀（T×d，可編輯）', items: [{ m: x, edit: true }] }]);

    // Layers 0..L-1
    for (let l = 0; l < L; l++) {
      const w = W.layers[l];
      const n1 = rms(x, w.g1);

      // Attn RMSNorm
      const tAttnNorm = stepsTpl.attn_rmsnorm;
      P(l, tAttnNorm.name, tAttnNorm.tex, tAttnNorm.desc, [
        { label: 'x（residual stream）', items: [{ m: x }] },
        { label: 'x̂ = RMSNorm(x)', items: [{ m: n1 }] }
      ]);

      // Q / K / V Projections
      const tq = tr(w.Wq), tk = tr(w.Wk), tv = tr(w.Wv);
      const Q = R(Hq).map(h => item(n1, tq.slice(h * hd, h * hd + hd), 'x̂', 'Wq'));
      const K = R(Hkv).map(g => item(n1, tk.slice(g * hd, g * hd + hd), 'x̂', 'Wk'));
      const V = R(Hkv).map(g => item(n1, tv.slice(g * hd, g * hd + hd), 'x̂', 'Wv'));
      const tQKV = stepsTpl.qkv_proj;
      P(l, tQKV.name, tQKV.tex, typeof tQKV.desc === 'function' ? tQKV.desc(Hq, Hkv, grp) : tQKV.desc, [
        { label: 'Q', by: 'q', items: Q },
        { label: 'K', by: 'kv', items: K },
        { label: 'V', by: 'kv', items: V }
      ]);

      // RoPE
      const Qr = Q.map(o => rope(o.m, hd, theta));
      const Kr = K.map(o => rope(o.m, hd, theta));
      const ang = R(T).map(t => R(hd / 2).map(i => (t * Math.pow(theta, (-2 * i) / hd) * 180 / Math.PI) % 360));
      const tRope = stepsTpl.rope;
      P(l, tRope.name, tRope.tex, tRope.desc, [
        { label: 'Q（RoPE 前）', by: 'q', items: Q },
        { label: "Q′（RoPE 後）", by: 'q', items: Qr.map(m => ({ m })) },
        { label: "K′（RoPE 後）", by: 'kv', items: Kr.map(m => ({ m })) },
        { label: '旋轉角 mθᵢ（度，mod 360）', items: [{ m: ang }] }
      ]);

      // Scores QK^T
      const S_arr = Qr.map((q, h) => item(q, Kr[Math.floor(h / grp)], "q′", "k′", Math.sqrt(hd)));
      const tScores = stepsTpl.scores;
      P(l, tScores.name, tScores.tex, tScores.desc, [{ label: 'S = Q′K′ᵀ/√d_h（mask 前）', by: 'q', items: S_arr }]);

      // Causal Mask + Softmax
      const Sm = S_arr.map(o => o.m.map((r, i) => r.map((v, j) => (j > i ? -Infinity : v))));
      const A = Sm.map(M =>
        M.map(r => {
          const mx = Math.max(...r);
          const e = r.map(v => Math.exp(v - mx));
          const s = e.reduce((a, b) => a + b);
          return e.map(v => v / s);
        })
      );
      const Ai = A.map((M, h) => ({
        m: M,
        prob: 1,
        micro: (i, j) => {
          const e = Sm[h][i].map(v => Math.exp(v));
          const s = e.reduce((a, b) => a + b);
          return `softmax 第 ${i} 列：exp(S)=[${e.slice(0, i + 1).map(f2).join(', ')}]，Σ=${f2(s)}，A[t${i}][${j}] = ${f2(e[j])}/${f2(s)} = <b>${f2(M[i][j])}</b>`;
        }
      }));
      const tSoftmax = stepsTpl.causal_softmax;
      P(l, tSoftmax.name, tSoftmax.tex, tSoftmax.desc, [
        { label: 'S（mask 後）', by: 'q', items: Sm.map(m => ({ m })) },
        { label: 'A = softmax(S)', by: 'q', items: Ai }
      ]);

      // Weighted sum A * V
      const O = A.map((a, h) => item(a, tr(V[Math.floor(h / grp)].m), 'A', 'V'));
      const tWeighted = stepsTpl.weighted_sum;
      P(l, tWeighted.name, tWeighted.tex, tWeighted.desc, [
        { label: 'A', by: 'q', items: Ai },
        { label: 'V（對應的 KV group）', by: 'kv', items: V },
        { label: 'O = A·V', by: 'q', items: O }
      ]);

      // o_proj
      const C = R(T).map(t => O.flatMap(o => o.m[t]));
      const ao = item(C, tr(w.Wo), 'C', 'Wo');
      const tOproj = stepsTpl.o_proj;
      P(l, tOproj.name, tOproj.tex, tOproj.desc, [
        { label: `C = concat(heads)（T×${Hq * hd}）`, items: [{ m: C }] },
        { label: 'attn_out = C·W_O', items: [ao] }
      ]);

      // Residual Add (Attn)
      const x2 = add(x, ao.m);
      S.ck.push({ x: x2, label: `L${l} attn` });
      const tResAttn = stepsTpl.residual_attn;
      P(
        l,
        tResAttn.name,
        tResAttn.tex,
        tResAttn.desc,
        [
          { label: 'x（前）', items: [{ m: x }] },
          { label: 'Δ = attn_out', items: [ao] },
          { label: 'x + Δ', items: [{ m: x2 }] }
        ],
        statsFn ? statsFn(x, ao.m, nrm, f2) : undefined
      );

      // MLP RMSNorm
      x = x2;
      const n2 = rms(x, w.g2);
      const tMlpNorm = stepsTpl.mlp_rmsnorm;
      P(l, tMlpNorm.name, tMlpNorm.tex, tMlpNorm.desc, [
        { label: 'x', items: [{ m: x }] },
        { label: 'x̂ = RMSNorm(x)', items: [{ m: n2 }] }
      ]);

      // Gate / Up Projections
      const G = item(n2, tr(w.Wg), 'x̂', 'Wg');
      const U = item(n2, tr(w.Wu), 'x̂', 'Wu');
      const tGateUp = stepsTpl.gate_up_proj;
      P(l, tGateUp.name, tGateUp.tex, typeof tGateUp.desc === 'function' ? tGateUp.desc(d, ffm) : tGateUp.desc, [
        { label: 'G = gate', items: [G] },
        { label: 'U = up', items: [U] }
      ]);

      // SwiGLU
      const Sg = G.m.map(r => r.map(silu));
      const Hh = Sg.map((r, i) => r.map((v, j) => v * U.m[i][j]));
      const tSwiglu = stepsTpl.swiglu;
      P(l, tSwiglu.name, tSwiglu.tex, tSwiglu.desc, [
        { label: 'silu(G)', items: [{ m: Sg }] },
        { label: 'U', items: [U] },
        { label: 'H = silu(G) ⊙ U', items: [{ m: Hh }] }
      ]);

      // Down Proj
      const Dn = item(Hh, tr(w.Wd), 'h', 'Wd');
      const tDown = stepsTpl.down_proj;
      P(l, tDown.name, tDown.tex, tDown.desc, [{ label: 'mlp_out = H·W_down', items: [Dn] }]);

      // Residual Add (MLP)
      const x3 = add(x, Dn.m);
      S.ck.push({ x: x3, label: `L${l} mlp` });
      const tResMlp = stepsTpl.residual_mlp;
      P(
        l,
        tResMlp.name,
        tResMlp.tex,
        tResMlp.desc,
        [
          { label: 'x（前）', items: [{ m: x }] },
          { label: 'Δ = mlp_out', items: [Dn] },
          { label: 'x + Δ', items: [{ m: x3 }] }
        ],
        statsFn ? statsFn(x, Dn.m, nrm, f2) : undefined
      );

      x = x3;
    }

    // Final RMSNorm
    const nf = rms(x, W.gf);
    S.ck.push({ x: nf, label: 'final' });
    const tFinal = stepsTpl.final_rmsnorm;
    P(L, tFinal.name, tFinal.tex, tFinal.desc, [
      { label: 'x', items: [{ m: x }] },
      { label: 'output', items: [{ m: nf }] }
    ]);
  }

  global.ToyLlama.model = {
    rng,
    gauss,
    rmat,
    tr,
    add,
    nrm,
    rms,
    f2,
    silu,
    ones,
    rope,
    item,
    forward
  };

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
