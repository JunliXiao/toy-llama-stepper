/**
 * Toy Llama Stepper - 視圖渲染與介面控制模組 (View & UI Rendering)
 */
(function (global) {
  'use strict';

  global.ToyLlama = global.ToyLlama || {};

  const $ = s => document.querySelector(s);

  /* ===== MathJax 序列化佇列與快取清理 ===== */
  let typesetQ = Promise.resolve();
  function typeset(el) {
    if (!el) return;
    if (!window.MathJax || !window.MathJax.typesetPromise) {
      if (window.MathJax?.startup?.promise) {
        window.MathJax.startup.promise.then(() => typeset(el));
      }
      return;
    }
    typesetQ = typesetQ
      .then(() => {
        if (window.MathJax.typesetClear) window.MathJax.typesetClear([el]);
        return window.MathJax.typesetPromise([el]);
      })
      .catch(console.error);
  }
  global.typeset = typeset;

  const curItem = tn => {
    const S = global.ToyLlama.state;
    return tn.items[tn.by === 'q' ? S.sel.h : tn.by === 'kv' ? Math.floor(S.sel.h / (S.cfg.Hq / S.cfg.Hkv)) : 0];
  };

  function matHTML(it, ti) {
    const S = global.ToyLlama.state;
    const f2 = global.ToyLlama.model.f2;
    const m = it.m, fin = m.flat().filter(isFinite), mx = Math.max(1e-9, ...fin.map(Math.abs));
    let h = '<div class="mw"><table class="mt"><tr><th></th>' + m[0].map((_, j) => `<th>${j}</th>`).join('') + '</tr>';
    m.forEach((r, i) => {
      h += `<tr><th>t${i}</th>`;
      r.forEach((v, j) => {
        const a = isFinite(v) ? (it.prob ? v : Math.min(1, Math.abs(v) / mx)) * 0.8 : 0;
        const c = it.prob || v >= 0 ? 'var(--pos)' : 'var(--neg)';
        const bg = `background:rgba(${c},${a})`;
        const on = S.sel.mi && S.sel.mi[0] === ti && S.sel.mi[1] === i && S.sel.mi[2] === j;
        h += it.edit
          ? `<td><input data-i="${i}" data-j="${j}" value="${v.toFixed(2)}" style="${bg}"></td>`
          : `<td class="${it.micro ? 'cl' : ''}${on ? ' on' : ''}" data-t="${ti}" data-i="${i}" data-j="${j}" style="${bg}">${f2(v)}</td>`;
      });
      h += '</tr>';
    });
    return h + '</table></div>';
  }

  function chart() {
    const S = global.ToyLlama.state;
    const { nrm, f2 } = global.ToyLlama.model;
    const w = 290, h = 130, n = S.ck.length;
    const mx = Math.max(...S.ck.flatMap(c => c.x.map(nrm))) * 1.1;
    const X = i => 25 + i * (w - 35) / (n - 1);
    const Y = v => h - 18 - (v / mx) * (h - 30);

    let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
    const ci = S.steps[S.sel.step] ? S.steps[S.sel.step].ci : 0;
    s += `<rect x="${X(ci) - 3}" y="6" width="6" height="${h - 24}" fill="rgba(var(--pos),.25)"/>`;
    for (let t = 0; t < S.cfg.T; t++) {
      s += `<polyline fill="none" stroke="hsl(${(t * 360) / S.cfg.T},65%,52%)" stroke-width="1.6" points="${S.ck.map((c, i) => X(i) + ',' + Y(nrm(c.x[t]))).join(' ')}"/>`;
    }
    s += S.ck.map((c, i) => `<text x="${X(i)}" y="${h - 4}" font-size="8" text-anchor="middle" fill="var(--mu)">${c.label.replace(/^L(\d) (.)/, '$1$2')}</text>`).join('');
    s += `<text x="2" y="12" font-size="9" fill="var(--mu)">${f2(mx)}</text></svg>`;
    return s;
  }

  function render() {
    const S = global.ToyLlama.state;
    const activeSideK = document.activeElement && document.activeElement.dataset?.k ? document.activeElement.dataset.k : null;
    const st = S.steps[S.sel.step];
    if (!st) return;

    const g = Math.floor(S.sel.h / (S.cfg.Hq / S.cfg.Hkv));
    $('#pos').textContent = `步驟 ${S.sel.step + 1} / ${S.steps.length}`;

    // 左側步驟清單
    let li = '', last = -2;
    S.steps.forEach((s, i) => {
      if (s.layer !== last) {
        last = s.layer;
        li += `<div class="lh">${s.layer < 0 ? 'Input' : s.layer === S.cfg.L ? 'Output' : 'Layer ' + s.layer}</div>`;
      }
      li += `<div class="si${i === S.sel.step ? ' a' : ''}" data-s="${i}">${s.name}</div>`;
    });
    $('#list').innerHTML = li;
    const a = $('#list .a');
    if (a) a.scrollIntoView({ block: 'nearest' });

    // 主畫面內容
    let m = `<h2>${st.layer >= 0 && st.layer < S.cfg.L ? 'Layer ' + st.layer + ' · ' : ''}${st.name}</h2><div class="card"><div id="fx">\\[${st.tex}\\]</div></div><div class="mu">${st.desc}</div>`;
    if (st.st) m += `<div class="card st">${st.st}</div>`;
    st.tensors.forEach((tn, ti) => {
      const it = curItem(tn);
      const lb = tn.by === 'q' ? ` — head ${S.sel.h}` : tn.by === 'kv' ? ` — KV group ${g}` : '';
      m += `<div class="tl">${tn.label}${lb}</div>${matHTML(it, ti)}`;
      if (S.sel.mi && S.sel.mi[0] === ti && it.micro) {
        m += `<div class="mic">${it.micro(S.sel.mi[1], S.sel.mi[2])}</div>`;
      }
    });
    if (st.tensors.some(t => curItem(t).micro)) {
      m += '<div class="mu" style="margin-top:6px">提示：點擊有底線游標的格子，可展開該格的逐項計算。</div>';
    }
    $('#main').innerHTML = m;

    // 恢復矩陣編輯格焦點
    if (S.pendingFocus) {
      const targetInput = $(`#main input[data-i="${S.pendingFocus.i}"][data-j="${S.pendingFocus.j}"]`);
      if (targetInput) {
        targetInput.focus();
        if (S.pendingFocus.select) targetInput.select();
      }
      S.pendingFocus = null;
    }

    // 右側資訊欄
    const c = S.cfg;
    const op = (arr, val) => arr.map(x => `<option${x === val ? ' selected' : ''}>${x}</option>`).join('');
    const mInfo = (global.TOY_MODELS && global.TOY_MODELS[S.sel.model]) || { name: '自訂模型', desc: '' };
    $('#side').innerHTML = `
      <h3>${mInfo.name}</h3>
      <div class="mu" style="margin-bottom:8px">${mInfo.desc || ''}</div>
      <div class="row"><span class="mu">維度數 (d_model)</span><span>${c.d}</span></div>
      <div class="row"><span class="mu">層數 (L)</span><span>${c.L}</span></div>
      <div class="row"><span class="mu">Q heads (Hq)</span><span style="display:inline-flex;align-items:center;gap:4px">${c.Hq} <span class="mu"> / 當前</span><select id="hs" style="padding:1px 5px">${[...Array(c.Hq).keys()].map(i => `<option value="${i}"${i === S.sel.h ? ' selected' : ''}>${i}</option>`).join('')}</select></span></div>
      <div class="row"><span class="mu">KV heads (Hkv)</span><span>${c.Hkv} (GQA ${c.Hq / c.Hkv}:1)</span></div>
      <div class="row"><span class="mu">head_dim (hd)</span><span>${c.hd}</span></div>
      <div class="row"><span class="mu">MLP 倍率 (ffm)</span><span>${c.ffm} (ff=${c.d * c.ffm})</span></div>
      <div class="row"><span class="mu">RoPE θ</span><span>${c.theta}</span></div>
      <h3 class="tip-title">輸入設定 (Input / Prompt) <span class="info-icon">ⓘ</span>
        <span class="tip-box">調整 seq_len 或 seed 會重設 X₀，不影響當前模型權重。</span>
      </h3>
      <div class="row">seq_len (T)<select data-k="T">${op([3, 4, 5, 6, 7, 8], c.T)}</select></div>
      <div class="row">seed<input data-k="seed" type="number" value="${c.seed}" style="width:90px"></div>
      <h3 class="tip-title">殘差流範數 ‖xₜ‖ <span class="info-icon">ⓘ</span>
        <span class="tip-box">每條線是一個 position；橫軸是各 block 寫入後的檢查點，淺色帶是目前步驟。</span>
      </h3>
      ${chart()}
      <h3>社群</h3>
      <div class="mu">
        <a href="https://github.com/JunliXiao/toy-llama-stepper" target="_blank" rel="noopener" title="GitHub Repository" class="social-icon-link">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 16 16" fill="currentColor" style="vertical-align: middle;">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8C0 11.54 2.29 14.53 5.47 15.59C5.87 15.66 6.02 15.42 6.02 15.21C6.02 15.02 6.01 14.39 6.01 13.72C4 14.09 3.48 13.23 3.32 12.78C3.23 12.55 2.84 11.84 2.5 11.65C2.22 11.5 1.82 11.13 2.49 11.12C3.12 11.11 3.57 11.7 3.72 11.94C4.44 13.15 5.59 12.81 6.05 12.6C6.12 12.08 6.33 11.73 6.56 11.53C4.78 11.33 2.92 10.64 2.92 7.58C2.92 6.71 3.23 5.99 3.74 5.43C3.66 5.23 3.38 4.41 3.82 3.31C3.82 3.31 4.49 3.1 6.02 4.13C6.66 3.95 7.34 3.86 8.02 3.86C8.7 3.86 9.38 3.95 10.02 4.13C11.55 3.09 12.22 3.31 12.22 3.31C12.66 4.41 12.38 5.23 12.3 5.43C12.81 5.99 13.12 6.7 13.12 7.58C13.12 10.65 11.25 11.33 9.47 11.53C9.76 11.78 10.01 12.26 10.01 13.01C10.01 14.08 10 14.94 10 15.21C10 15.42 10.15 15.67 10.55 15.59C13.71 14.53 16 11.53 16 8C16 3.58 12.42 0 8 0Z" />
          </svg>
        </a>
        <a href="https://www.threads.com/@xiao.junli" target="_blank" rel="noopener" title="Threads" class="social-icon-link">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 192 192" fill="currentColor" style="vertical-align: middle;">
            <path d="M141.537 88.9883C140.71 88.5919 139.87 88.2104 139.019 87.8451C137.537 60.5382 122.616 44.905 97.5619 44.745C97.4484 44.7443 97.3355 44.7443 97.222 44.7443C82.2364 44.7443 69.7731 51.1409 62.102 62.7807L75.881 72.2328C81.6116 63.5383 90.6052 61.6848 97.2286 61.6848C97.3051 61.6848 97.3819 61.6848 97.4576 61.6855C105.707 61.7381 111.932 64.1366 115.961 68.814C118.893 72.2193 120.854 76.925 121.825 82.8638C114.511 81.6207 106.601 81.2385 98.145 81.7233C74.3247 83.0954 59.0111 96.9879 60.0396 116.292C60.5615 126.084 65.4397 134.508 73.775 140.011C80.8224 144.663 89.899 146.938 99.3323 146.423C111.79 145.74 121.563 140.987 128.381 132.296C133.559 125.696 136.834 117.143 138.28 106.366C144.217 109.949 148.617 114.664 151.047 120.332C155.179 129.967 155.42 145.8 142.501 158.708C131.182 170.016 117.576 174.908 97.0135 175.059C74.2042 174.89 56.9538 167.575 45.7381 153.317C35.2355 139.966 29.8077 120.682 29.6052 96C29.8077 71.3178 35.2355 52.0336 45.7381 38.6827C56.9538 24.4249 74.2039 17.11 97.0132 16.9405C119.988 17.1113 137.539 24.4614 149.184 38.788C154.894 45.8136 159.199 54.6488 162.037 64.9503L178.184 60.6422C174.744 47.9622 169.331 37.0357 161.965 27.974C147.036 9.60668 125.202 0.195148 97.0695 0H96.9569C68.8816 0.19447 47.2921 9.6418 32.7883 28.0793C19.8819 44.4864 13.2244 67.3157 13.0007 95.9325L13 96L13.0007 96.0675C13.2244 124.684 19.8819 147.514 32.7883 163.921C47.2921 182.358 68.8816 191.806 96.9569 192H97.0695C122.03 191.827 139.624 185.292 154.118 170.811C173.081 151.866 172.51 128.119 166.26 113.541C161.776 103.087 153.227 94.5962 141.537 88.9883ZM98.4405 129.507C88.0005 130.095 77.1544 125.409 76.6196 115.372C76.2232 107.93 81.9158 99.626 99.0812 98.6368C101.047 98.5234 102.976 98.468 104.871 98.468C111.106 98.468 116.939 99.0737 122.242 100.233C120.264 124.935 108.662 128.946 98.4405 129.507Z" />
          </svg>
        </a>
      </div>
    `;

    if (activeSideK) {
      const targetSideEl = $(`#side [data-k="${activeSideK}"]`);
      if (targetSideEl) targetSideEl.focus();
    }

    const fx = $('#fx');
    if (fx) typeset(fx);
  }

  function fillHs() {
    const S = global.ToyLlama.state;
    if ($('#hs')) {
      $('#hs').innerHTML = [...Array(S.cfg.Hq).keys()]
        .map(i => `<option value="${i}"${i === S.sel.h ? ' selected' : ''}>${i}</option>`)
        .join('');
    }
  }

  function fillMs() {
    const S = global.ToyLlama.state;
    const models = global.TOY_MODELS || {};
    const keys = Object.keys(models);
    if (keys.length && $('#ms')) {
      $('#ms').innerHTML = keys
        .map(k => `<option value="${k}"${k === S.sel.model ? ' selected' : ''}>${models[k].name}</option>`)
        .join('');
    }
  }

  const toggleModal = open => {
    const el = $('#modal-kb');
    if (!el) return;
    if (open === undefined) el.classList.toggle('open');
    else if (open) el.classList.add('open');
    else el.classList.remove('open');
  };

  const toggleL = () => {
    const isHidden = $('#app').classList.toggle('hide-l');
    $('#tl').classList.toggle('toggled', isHidden);
    $('#tl').title = isHidden ? '展開左側步驟清單 (快捷鍵: [)' : '收合左側步驟清單 (快捷鍵: [)';
  };

  const toggleR = () => {
    const isHidden = $('#app').classList.toggle('hide-r');
    $('#tr').classList.toggle('toggled', isHidden);
    $('#tr').title = isHidden ? '展開右側資訊欄 (快捷鍵: ])' : '收合右側資訊欄 (快捷鍵: ])';
  };

  global.ToyLlama.view = {
    $,
    typeset,
    matHTML,
    chart,
    render,
    fillHs,
    fillMs,
    toggleModal,
    toggleL,
    toggleR
  };

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
