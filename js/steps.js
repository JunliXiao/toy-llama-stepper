/**
 * Toy Llama Stepper - 步驟定義與文案字典 (Steps Metadata & Templates)
 * 供模型計算與多國語言擴充 (i18n) 使用
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
      input_x0: {
        name: 'Input X₀',
        tex: String.raw`X_0\in\mathbb{R}^{T\times d}`,
        desc: '直接把 X₀ 當作 embedding 之後的 residual stream（殘差流）：每一列是一個 position 的向量。你可以直接編輯下面任何一格，後面所有步驟會即時重算。'
      },
      attn_rmsnorm: {
        name: 'Attn RMSNorm',
        tex: RMS,
        desc: 'pre-norm：每個 position 各自除以自己的 RMS，再乘增益 g（此處 g=1）。只改變向量長度，不改變方向；注意 residual stream 本身沒有被改動，送進 attention 的只有 x̂。'
      },
      qkv_proj: {
        name: 'Q / K / V 投影',
        tex: String.raw`Q_h=\hat X W_Q^{(h)},\quad K_g=\hat X W_K^{(g)},\quad V_g=\hat X W_V^{(g)},\quad g=\lfloor h/(H_q/H_{kv})\rfloor`,
        desc: (Hq, Hkv, grp) => `GQA：${Hq} 個 Q head 共用 ${Hkv} 組 K/V，每 ${grp} 個 Q head 共用一組。Q 決定「我想找什麼」，K 決定「我能被怎麼找到」，V 是「被讀走的內容」。點任一格可展開成逐項乘加。`
      },
      rope: {
        name: 'RoPE',
        tex: String.raw`\begin{pmatrix}q'_i\\ q'_{i+d_h/2}\end{pmatrix}=\begin{pmatrix}\cos m\theta_i&-\sin m\theta_i\\ \sin m\theta_i&\cos m\theta_i\end{pmatrix}\begin{pmatrix}q_i\\ q_{i+d_h/2}\end{pmatrix},\quad\theta_i=\Theta^{-2i/d_h}`,
        desc: '只旋轉 Q 與 K（V 不動）。位置 m 的向量在第 i 個二維平面轉 mθᵢ，向量長度不變，但 Q·K 之後只依賴相對距離。θᵢ 隨 i 變小：前面的維度轉得快（高頻），後面轉得慢（低頻）。'
      },
      scores: {
        name: 'Scores QKᵀ',
        tex: String.raw`S_h=\frac{Q'_h\,{K'_g}^{\top}}{\sqrt{d_h}}`,
        desc: 'S[i][j] 是 position i 的 query 與 position j 的 key 的內積（除以 √d_h 穩定尺度）。此時還沒有 mask，上三角的值之後會被丟掉。'
      },
      causal_softmax: {
        name: 'Causal mask + Softmax',
        tex: String.raw`A_{ij}=\frac{e^{S_{ij}}}{\sum_{k\le i}e^{S_{ik}}}\ (j\le i),\qquad A_{ij}=0\ (j>i)`,
        desc: '因果遮罩：position i 只能看 j≤i，被遮的位置設為 −∞，softmax 後為 0。每一列加總為 1，A[i][j] 就是 position i 從 position j 讀多少。'
      },
      weighted_sum: {
        name: '加權和 A·V',
        tex: String.raw`O_h=A_h\,V_g`,
        desc: '每個 position 把「可見位置的 V」按注意力權重混合起來，這就是 attention 唯一跨 position 搬運資訊的步驟。'
      },
      o_proj: {
        name: 'o_proj',
        tex: String.raw`\mathrm{attn}=\big[O_1\,\|\,\cdots\,\|\,O_{H_q}\big]\,W_O`,
        desc: '把所有 head 的輸出接起來，再用 W_O 投影回 d_model 維。等價於「每個 head 各自經過 W_O 的一個切片，再相加寫入 residual stream」。'
      },
      residual_attn: {
        name: 'Residual 加回（attn）',
        tex: String.raw`x\leftarrow x+\mathrm{Attn}(\mathrm{RMSNorm}(x))`,
        desc: 'attention 的輸出以「加法」寫回 residual stream。下表是每個 position 的寫入量 ‖Δ‖ 與寫入方向和原向量的夾角。'
      },
      mlp_rmsnorm: {
        name: 'MLP RMSNorm',
        tex: RMS,
        desc: 'MLP 前的第二個 pre-norm。'
      },
      gate_up_proj: {
        name: 'gate / up 投影',
        tex: String.raw`G=\hat X W_{gate},\qquad U=\hat X W_{up}`,
        desc: (d, ffm) => `兩條平行的線性投影，維度從 ${d} 擴到 ${d * ffm}。MLP 對每個 position 獨立作用，不跨 token。`
      },
      swiglu: {
        name: 'SwiGLU',
        tex: String.raw`H=\mathrm{silu}(G)\odot U,\qquad \mathrm{silu}(z)=z\,\sigma(z)=\frac{z}{1+e^{-z}}`,
        desc: 'silu(G) 當作「閘門」，逐元素去調制 U：閘門接近 0 的神經元被關掉，正值較大的被放行。'
      },
      down_proj: {
        name: 'down_proj',
        tex: String.raw`\mathrm{mlp}=H\,W_{down}`,
        desc: '投影回 d_model。從 mech interp 角度，可視為「ff 個神經元各自帶著 W_down 的一列，按啟動量加權寫入 residual stream」。'
      },
      residual_mlp: {
        name: 'Residual 加回（MLP）',
        tex: String.raw`x\leftarrow x+\mathrm{MLP}(\mathrm{RMSNorm}(x))`,
        desc: 'MLP 的輸出加回 residual stream，完成一個 block。'
      },
      final_rmsnorm: {
        name: 'Final RMSNorm',
        tex: RMS,
        desc: '最後一層之後的 norm。真實模型在這之後接 unembed 變成 logits，此處省略，輸出就是這個矩陣。'
      }
    }
  };

  global.ToyLlama.steps = stepsMeta;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
