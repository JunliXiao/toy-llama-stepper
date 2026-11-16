/**
 * Toy Llama Stepper - 步驟數學公式與統計模板 (Pure Math Formulas & Stats)
 * 自然語言名稱與教學說明已集中抽離至 js/i18n.js
 */
(function (global) {
  'use strict';

  global.ToyLlama = global.ToyLlama || {};

  const RMS = String.raw`\hat{x}_t=\frac{x_t}{\sqrt{\tfrac1d\sum_{i=1}^{d}x_{t,i}^2+\varepsilon}}\odot g`;

  function stats(x, dx, nrm, f2) {
    return '<table class="mt"><tr><th></th><th>‖x‖</th><th>‖Δ‖</th><th>cos(x,Δ)</th><th>‖x+Δ‖</th></tr>' +
      x.map((r, t) => {
        const a = nrm(r), b = nrm(dx[t]);
        const c = r.reduce((s, v, i) => s + v * dx[t][i], 0) / (a * b + 1e-12);
        const y = nrm(r.map((v, i) => v + dx[t][i]));
        return `<tr><th>t${t}</th><td>${f2(a)}</td><td>${f2(b)}</td><td>${f2(c)}</td><td>${f2(y)}</td></tr>`;
      }).join('') + '</table>';
  }

  const stepsMeta = {
    RMS,
    stats,
    templates: {
      input_x0: { tex: String.raw`X_0\in\mathbb{R}^{T\times d}` },
      attn_rmsnorm: { tex: RMS },
      qkv_proj: { tex: String.raw`Q_h=\hat X W_Q^{(h)},\quad K_g=\hat X W_K^{(g)},\quad V_g=\hat X W_V^{(g)},\quad g=\lfloor h/(H_q/H_{kv})\rfloor` },
      rope: { tex: String.raw`\begin{pmatrix}q'_i\\ q'_{i+d_h/2}\end{pmatrix}=\begin{pmatrix}\cos m\theta_i&-\sin m\theta_i\\ \sin m\theta_i&\cos m\theta_i\end{pmatrix}\begin{pmatrix}q_i\\ q_{i+d_h/2}\end{pmatrix},\quad\theta_i=\Theta^{-2i/d_h}` },
      scores: { tex: String.raw`S_h=\frac{Q'_h\,{K'_g}^{\top}}{\sqrt{d_h}}` },
      causal_softmax: { tex: String.raw`A_{ij}=\frac{e^{S_{ij}}}{\sum_{k\le i}e^{S_{ik}}}\ (j\le i),\qquad A_{ij}=0\ (j>i)` },
      weighted_sum: { tex: String.raw`O_h=A_h\,V_g` },
      o_proj: { tex: String.raw`\mathrm{attn}=\big[O_1\,\|\,\cdots\,\|\,O_{H_q}\big]\,W_O` },
      residual_attn: { tex: String.raw`x\leftarrow x+\mathrm{Attn}(\mathrm{RMSNorm}(x))` },
      mlp_rmsnorm: { tex: RMS },
      gate_up_proj: { tex: String.raw`G=\hat X W_{gate},\qquad U=\hat X W_{up}` },
      swiglu: { tex: String.raw`H=\mathrm{silu}(G)\odot U,\qquad \mathrm{silu}(z)=z\,\sigma(z)=\frac{z}{1+e^{-z}}` },
      down_proj: { tex: String.raw`\mathrm{mlp}=H\,W_{down}` },
      residual_mlp: { tex: String.raw`x\leftarrow x+\mathrm{MLP}(\mathrm{RMSNorm}(x))` },
      final_rmsnorm: { tex: RMS }
    }
  };

  global.ToyLlama.steps = stepsMeta;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);