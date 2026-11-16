/**
 * Toy Llama Stepper - 多國語系字典與查詢模組 (Internationalization Dictionary)
 * 支援繁體中文 (zh-Hant) 與英文 (en)，純記憶體運行、無狀態相依
 */
(function (global) {
  'use strict';

  global.ToyLlama = global.ToyLlama || {};

  const dict = {
    'zh-Hant': {
      ui: {
        doc_title: 'Toy Llama Block Stepper - LLM 逐步計算與線性代數視覺化工具',
        first_step: '跳至第一步 (快捷鍵: Ctrl+←)',
        prev: '◀ 上一步',
        next: '下一步 ▶',
        last_step: '跳至最後一步 (快捷鍵: Ctrl+→)',
        model_label: '模型',
        toggle_l_hide: '收合左側步驟清單 (快捷鍵: [)',
        toggle_l_show: '展開左側步驟清單 (快捷鍵: [)',
        toggle_r_hide: '收合右側資訊欄 (快捷鍵: ])',
        toggle_r_show: '展開右側資訊欄 (快捷鍵: ])',
        lang_btn: 'EN',
        lang_title: 'Switch to English / 切換為英文',
        hk_title: '鍵盤快捷鍵指南 (快捷鍵: ?)',
        theme_title: '切換深淺色主題',
        step_pos: (cur, total) => `步驟 ${cur} / ${total}`,
        head_badge: h => ` — head ${h}`,
        kv_badge: g => ` — KV group ${g}`,
        micro_hint: '提示：點擊有底線游標的格子，可展開該格的逐項計算。',
        custom_model: '自訂模型',
        dim_label: '維度數 (d_model)',
        layers_label: '層數 (L)',
        q_heads_label: 'Q heads (Hq)',
        current_head: '當前',
        kv_heads_label: 'KV heads (Hkv)',
        head_dim_label: 'head_dim (hd)',
        mlp_mult_label: 'MLP 倍率 (ffm)',
        rope_theta_label: 'RoPE θ',
        input_sec_title: '輸入設定 (Input / Prompt)',
        input_sec_tip: '調整 seq_len 或 seed 會重設 X₀，不影響當前模型權重。',
        norm_sec_title: '殘差流範數 ‖xₜ‖',
        norm_sec_tip: '每條線是一個 position；橫軸是各 block 寫入後的檢查點，淺色帶是目前步驟。',
        social_sec_title: '社群'
      },
      modal: {
        title: '⌨️ 鍵盤快捷鍵指南',
        sec_nav: '導航操作',
        nav_step: '上一步 / 下一步',
        nav_end: '跳至第一步 (X₀) / 最後一步',
        nav_layer: '跨 Layer 同階段移動',
        sec_view: '介面與觀察維度',
        view_sidebars: '收合/展開左側欄 / 右側欄',
        view_head: '循環切換上一個 / 下一個 Q head',
        footer: '提示：按 <span class="kbd">?</span> 開關此指南，按 <span class="kbd">Esc</span> 關閉。'
      },
      tensors: {
        x0_edit: 'X₀（T×d，可編輯）',
        x_res: 'x（residual stream）',
        x_norm: 'x̂ = RMSNorm(x)',
        q_before_rope: 'Q（RoPE 前）',
        q_after_rope: 'Q′（RoPE 後）',
        k_after_rope: 'K′（RoPE 後）',
        rope_angles: '旋轉角 mθᵢ（度，mod 360）',
        scores_pre_mask: 'S = Q′K′ᵀ/√d_h（mask 前）',
        scores_post_mask: 'S（mask 後）',
        v_kv_group: 'V（對應的 KV group）',
        concat_heads: dim => `C = concat(heads)（T×${dim}）`,
        x_prev: 'x（前）',
        softmax_micro: (row, expList, sum, col, val, prob) =>
          `softmax 第 ${row} 列：exp(S)=[${expList}]，Σ=${sum}，A[t${row}][${col}] = ${val}/${sum} = <b>${prob}</b>`
      },
      steps: {
        input_x0: {
          name: 'Input X₀',
          desc: '直接把 X₀ 當作 embedding 之後的 residual stream（殘差流）：每一列是一個 position 的向量。你可以直接編輯下面任何一格，後面所有步驟會即時重算。'
        },
        attn_rmsnorm: {
          name: 'Attn RMSNorm',
          desc: 'pre-norm：每個 position 各自除以自己的 RMS，再乘增益 g（此處 g=1）。只改變向量長度，不改變方向；注意 residual stream 本身沒有被改動，送進 attention 的只有 x̂。'
        },
        qkv_proj: {
          name: 'Q / K / V 投影',
          desc: (Hq, Hkv, grp) => `GQA：${Hq} 個 Q head 共用 ${Hkv} 組 K/V，每 ${grp} 個 Q head 共用一組。Q 決定「我想找什麼」，K 決定「我能被怎麼找到」，V 是「被讀走的內容」。點任一格可展開成逐項乘加。`
        },
        rope: {
          name: 'RoPE',
          desc: '只旋轉 Q 與 K（V 不動）。位置 m 的向量在第 i 個二維平面轉 mθᵢ，向量長度不變，但 Q·K 之後只依賴相對距離。θᵢ 隨 i 變小：前面的維度轉得快（高頻），後面轉得慢（低頻）。'
        },
        scores: {
          name: 'Scores QKᵀ',
          desc: 'S[i][j] 是 position i 的 query 與 position j 的 key 的內積（除以 √d_h 穩定尺度）。此時還沒有 mask，上三角的值之後會被丟掉。'
        },
        causal_softmax: {
          name: 'Causal mask + Softmax',
          desc: '因果遮罩：position i 只能看 j≤i，被遮的位置設為 −∞，softmax 後為 0。每一列加總為 1，A[i][j] 就是 position i 從 position j 讀多少。'
        },
        weighted_sum: {
          name: '加權和 A·V',
          desc: '每個 position 把「可見位置的 V」按注意力權重混合起來，這就是 attention 唯一跨 position 搬運資訊的步驟。'
        },
        o_proj: {
          name: 'o_proj',
          desc: '把所有 head 的輸出接起來，再用 W_O 投影回 d_model 維。等價於「每個 head 各自經過 W_O 的一個切片，再相加寫入 residual stream」。'
        },
        residual_attn: {
          name: 'Residual 加回（attn）',
          desc: 'attention 的輸出以「加法」寫回 residual stream。下表是每個 position 的寫入量 ‖Δ‖ 與寫入方向和原向量的夾角。'
        },
        mlp_rmsnorm: {
          name: 'MLP RMSNorm',
          desc: 'MLP 前的第二個 pre-norm。'
        },
        gate_up_proj: {
          name: 'gate / up 投影',
          desc: (d, ffm) => `兩條平行的線性投影，維度從 ${d} 擴到 ${d * ffm}。MLP 對每個 position 獨立作用，不跨 token。`
        },
        swiglu: {
          name: 'SwiGLU',
          desc: 'silu(G) 當作「閘門」，逐元素去調制 U：閘門接近 0 的神經元被關掉，正值較大的被放行。'
        },
        down_proj: {
          name: 'down_proj',
          desc: '投影回 d_model。從 mech interp 角度，可視為「ff 個神經元各自帶著 W_down 的一列，按啟動量加權寫入 residual stream」。'
        },
        residual_mlp: {
          name: 'Residual 加回（MLP）',
          desc: 'MLP 的輸出加回 residual stream，完成一個 block。'
        },
        final_rmsnorm: {
          name: 'Final RMSNorm',
          desc: '最後一層之後的 norm。真實模型在這之後接 unembed 變成 logits，此處省略，輸出就是這個矩陣。'
        }
      },
      mobile_modal: {
        title: '💻 建議使用桌面版瀏覽器',
        desc: '本工具主要針對桌面系統設計。您在行動裝置上仍可繼續瀏覽，但建議改用電腦開啟以獲得最佳體驗。',
        ok: '我知道了，繼續使用'
      },
      models: {
        baseline: {
          name: '基準模型：Random Baseline, Seed 7',
          desc: '標準隨機初始化模型（Seed 7）。注意力與前饋特徵呈均勻高斯分佈，展示一般 Transformer 數值流動。'
        },
        prev_token: {
          name: '機制模型：Previous-Token Head',
          desc: '將 Layer 0 的 Head 0 設計為專注於「前一個 Token (t-1)」的注意力頭。請切換至「Causal mask + Softmax」步驟並觀察 Head 0 的次對角線注意力高權重。'
        },
        swiglu_gate: {
          name: '機制模型：SwiGLU 特徵開關',
          desc: '將 Layer 0 的 MLP 神經元特化為「特徵檢測器與記憶寫入」。第 0 號神經元專注檢測「維度 0 為正值」並飽和放行；第 1 號神經元檢測「維度 0 為負值」。請在「SwiGLU」與「down_proj」步驟觀察閘門 (SiLU) 的精準篩選。'
        }
      }
    },

    'en': {
      ui: {
        doc_title: 'Toy Llama Block Stepper - Step-by-Step LLM & Linear Algebra Visualizer',
        first_step: 'Jump to first step (Shortcut: Ctrl+←)',
        prev: '◀ Prev',
        next: 'Next ▶',
        last_step: 'Jump to last step (Shortcut: Ctrl+→)',
        model_label: 'Model',
        toggle_l_hide: 'Collapse left step list (Shortcut: [)',
        toggle_l_show: 'Expand left step list (Shortcut: [)',
        toggle_r_hide: 'Collapse right inspector (Shortcut: ])',
        toggle_r_show: 'Expand right inspector (Shortcut: ])',
        lang_btn: '繁中',
        lang_title: '切換為繁體中文 / Switch to Traditional Chinese',
        hk_title: 'Keyboard shortcuts guide (Shortcut: ?)',
        theme_title: 'Toggle light/dark theme',
        step_pos: (cur, total) => `Step ${cur} / ${total}`,
        head_badge: h => ` — head ${h}`,
        kv_badge: g => ` — KV group ${g}`,
        micro_hint: 'Tip: Click any cell with a pointer cursor to inspect its term-by-term arithmetic.',
        custom_model: 'Custom Model',
        dim_label: 'Dimension (d_model)',
        layers_label: 'Layers (L)',
        q_heads_label: 'Q heads (Hq)',
        current_head: 'current',
        kv_heads_label: 'KV heads (Hkv)',
        head_dim_label: 'head_dim (hd)',
        mlp_mult_label: 'MLP mult (ffm)',
        rope_theta_label: 'RoPE θ',
        input_sec_title: 'Input Settings',
        input_sec_tip: 'Changing seq_len or seed resets X₀ without modifying model weights.',
        norm_sec_title: 'Residual Stream Norm ‖xₜ‖',
        norm_sec_tip: 'Each line is a token position; the x-axis shows checkpoints after each block, and the shaded band marks the current step.',
        social_sec_title: 'Links'
      },
      modal: {
        title: '⌨️ Keyboard Shortcuts',
        sec_nav: 'Navigation',
        nav_step: 'Previous / Next step',
        nav_end: 'Jump to first (X₀) / last step',
        nav_layer: 'Move across layers at the same stage',
        sec_view: 'View & Dimensions',
        view_sidebars: 'Toggle left / right sidebar',
        view_head: 'Cycle previous / next Q head',
        footer: 'Tip: Press <span class="kbd">?</span> to toggle this guide, or <span class="kbd">Esc</span> to close.'
      },
      tensors: {
        x0_edit: 'X₀ (T×d, editable)',
        x_res: 'x (residual stream)',
        x_norm: 'x̂ = RMSNorm(x)',
        q_before_rope: 'Q (before RoPE)',
        q_after_rope: 'Q′ (after RoPE)',
        k_after_rope: 'K′ (after RoPE)',
        rope_angles: 'Rotation angle mθᵢ (deg, mod 360)',
        scores_pre_mask: 'S = Q′K′ᵀ/√d_h (pre-mask)',
        scores_post_mask: 'S (masked)',
        v_kv_group: 'V (corresponding KV group)',
        concat_heads: dim => `C = concat(heads) (T×${dim})`,
        x_prev: 'x (before)',
        softmax_micro: (row, expList, sum, col, val, prob) =>
          `softmax row ${row}: exp(S)=[${expList}], Σ=${sum}, A[t${row}][${col}] = ${val}/${sum} = <b>${prob}</b>`
      },
      steps: {
        input_x0: {
          name: 'Input X₀',
          desc: 'Treat X₀ directly as the post-embedding residual stream: each row is a token position vector. Edit any cell below and all subsequent steps will recompute immediately.'
        },
        attn_rmsnorm: {
          name: 'Attn RMSNorm',
          desc: 'Pre-norm: each position vector is divided by its own RMS and scaled by gain g (here g=1). This rescales vector magnitude without changing direction; the residual stream itself remains untouched—only x̂ enters attention.'
        },
        qkv_proj: {
          name: 'Q / K / V Projections',
          desc: (Hq, Hkv, grp) => `GQA: ${Hq} Q heads share ${Hkv} K/V groups (${grp} Q heads per group). Q defines "what I am looking for", K defines "how I can be matched", and V is "the content to be read". Click any cell for term-by-term dot products.`
        },
        rope: {
          name: 'RoPE',
          desc: 'Rotates only Q and K (leaving V unchanged). Position m is rotated by mθᵢ in the i-th 2D plane, preserving vector norms while making Q·K depend strictly on relative distance. θᵢ decreases with i: early dimensions rotate fast (high frequency), later ones slowly (low frequency).'
        },
        scores: {
          name: 'Scores QKᵀ',
          desc: 'S[i][j] is the dot product between position i’s query and position j’s key (scaled by 1/√d_h for stability). No causal mask is applied yet; upper-triangular entries will be masked next.'
        },
        causal_softmax: {
          name: 'Causal mask + Softmax',
          desc: 'Causal mask: position i can only attend to j≤i. Future positions (j>i) are set to −∞, becoming 0 after softmax. Each row sums to 1, where A[i][j] is how much position i reads from position j.'
        },
        weighted_sum: {
          name: 'Weighted Sum A·V',
          desc: 'Each position blends the value vectors V of visible tokens weighted by attention probabilities A. This is the only operation in attention that moves information across token positions.'
        },
        o_proj: {
          name: 'o_proj',
          desc: 'Concatenate all head outputs and project back to d_model via W_O. Equivalently, each head passes through its own slice of W_O and adds its result into the residual stream.'
        },
        residual_attn: {
          name: 'Residual Add (Attn)',
          desc: 'The attention output is added back into the residual stream. The table below reports the update norm ‖Δ‖ and cosine alignment with the incoming vector x at each position.'
        },
        mlp_rmsnorm: {
          name: 'MLP RMSNorm',
          desc: 'The second pre-norm before the MLP sub-layer.'
        },
        gate_up_proj: {
          name: 'gate / up Projections',
          desc: (d, ffm) => `Two parallel linear projections expanding dimension from ${d} to ${d * ffm}. The MLP acts independently on each token position without cross-token mixing.`
        },
        swiglu: {
          name: 'SwiGLU',
          desc: 'silu(G) acts as a soft gate modulating U elementwise: neurons with gates near 0 are suppressed, while strongly positive gates pass signal through.'
        },
        down_proj: {
          name: 'down_proj',
          desc: 'Project back to d_model. From a mechanistic interpretability view, each of the ff neurons scales its corresponding row of W_down by its activation and writes into the residual stream.'
        },
        residual_mlp: {
          name: 'Residual Add (MLP)',
          desc: 'The MLP output is added back into the residual stream, completing one Transformer block.'
        },
        final_rmsnorm: {
          name: 'Final RMSNorm',
          desc: 'Final normalization after the last layer. In a full LLM, an unembedding matrix follows to produce logits; here it is omitted, and this matrix is the final output.'
        }
      },
      mobile_modal: {
        title: '💻 Desktop Browser Recommended',
        desc: 'This tool is designed primarily for desktop screens. You can still continue on mobile, though a desktop device is recommended for the best experience.',
        ok: 'Got it, continue'
      },
      models: {
        baseline: {
          name: 'Baseline: Random Baseline, Seed 7',
          desc: 'Standard randomly initialized weights (Seed 7). Attention and MLP activations follow uniform Gaussian distributions, illustrating typical Transformer value flow.'
        },
        prev_token: {
          name: 'Mechanistic: Previous-Token Head',
          desc: 'Layer 0 Head 0 is engineered to attend specifically to the preceding token (t-1). Jump to "Causal mask + Softmax" and inspect Head 0 to observe strong sub-diagonal attention weights.'
        },
        swiglu_gate: {
          name: 'Mechanistic: SwiGLU Feature Gate',
          desc: 'Layer 0 MLP neurons act as feature detectors and key-value writers: Neuron 0 fires when Dim 0 is positive, while Neuron 1 fires when Dim 0 is negative. Inspect "SwiGLU" and "down_proj" to see selective SiLU gating.'
        }
      }
    }
  };

  function t(path, ...args) {
    const lang = global.ToyLlama.state?.sel?.lang || 'en';
    const resolve = obj => path.split('.').reduce((acc, k) => (acc && acc[k] !== undefined ? acc[k] : undefined), obj);
    const val = resolve(dict[lang]) ?? resolve(dict['en']) ?? path;
    return typeof val === 'function' ? val(...args) : val;
  }

  global.ToyLlama.i18n = { dict, t };

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);