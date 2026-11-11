/**
 * Toy Llama Stepper - 應用程式入口與事件監聽 (App Bootstrap & Event Listeners)
 */
(function (global) {
  'use strict';

  const { state, view } = global.ToyLlama;
  const $ = view.$;

  // 步驟導覽按鈕
  $('#b0').onclick = () => state.go(0);
  $('#b1').onclick = () => state.go(state.steps.length - 1);
  $('#bp').onclick = () => state.go(state.sel.step - 1);
  $('#bn').onclick = () => state.go(state.sel.step + 1);

  // 主題切換 (深淺模式)
  $('#th').onclick = () => {
    const r = document.documentElement;
    const isDark = r.dataset.theme !== 'dark';
    r.dataset.theme = isDark ? 'dark' : 'light';

    const iconEl = $('#favicon');
    if (iconEl) {
      iconEl.href = isDark ? './resources/icon-dark.svg' : './resources/icon-light.svg';
    }
  };

  // 快捷鍵指南 Modal
  $('#hk').onclick = () => view.toggleModal(true);
  $('#modal-close').onclick = () => view.toggleModal(false);
  $('#modal-kb').onclick = e => {
    if (e.target === $('#modal-kb')) view.toggleModal(false);
  };

  // 側邊欄收合按鈕
  $('#tl').onclick = view.toggleL;
  $('#tr').onclick = view.toggleR;

  // 左側步驟清單點擊
  $('#list').onclick = e => {
    const s = e.target.closest('.si');
    if (s) state.go(+s.dataset.s);
  };

  // 主視圖矩陣互動
  $('#main').onclick = e => {
    const td = e.target.closest('td.cl');
    if (td) {
      state.sel.mi = [+td.dataset.t, +td.dataset.i, +td.dataset.j];
      view.render();
    }
  };

  $('#main').addEventListener('keydown', e => {
    const t = e.target;
    if (t.dataset.i == null) return;
    const i = +t.dataset.i, j = +t.dataset.j;
    const d = state.cfg.d, T = state.cfg.T;

    if (e.code === 'Tab') {
      e.preventDefault();
      let nextI = i, nextJ = j + (e.shiftKey ? -1 : 1);
      if (nextJ >= d) { nextI++; nextJ = 0; }
      else if (nextJ < 0) { nextI--; nextJ = d - 1; }

      state.X0[i][j] = parseFloat(t.value) || 0;
      if (nextI >= 0 && nextI < T) {
        state.pendingFocus = { i: nextI, j: nextJ, select: true };
      }
      state.recompute(false);
    } else if (e.code === 'Enter' || e.code === 'NumpadEnter') {
      e.preventDefault();
      state.pendingFocus = null;
      state.X0[i][j] = parseFloat(t.value) || 0;
      t.blur();
      state.recompute(false);
    }
  });

  $('#main').onchange = e => {
    const t = e.target;
    if (t.dataset.i != null) {
      const val = parseFloat(t.value) || 0;
      if (state.X0[+t.dataset.i][+t.dataset.j] !== val) {
        state.X0[+t.dataset.i][+t.dataset.j] = val;
        state.recompute(false);
      }
    }
  };

  // 模型切換
  $('#ms').onchange = e => {
    state.loadModel(e.target.value, false);
  };

  // 右側欄參數調整
  $('#side').onchange = e => {
    const t = e.target;
    if (t.id === 'hs') {
      state.sel.h = +t.value;
      state.sel.mi = null;
      view.render();
      return;
    }
    const k = t.dataset.k;
    if (!k) return;
    state.cfg[k] = +t.value;
    state.initInput(true);
    state.recompute(false);
  };

  // 全域鍵盤快捷鍵
  document.addEventListener('keydown', e => {
    // 1. 若當前焦點在任何輸入元件，不觸發全域快捷鍵
    if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;

    // 2. 若處於輸入法組字狀態 (IME composing)，不觸發
    if (e.isComposing || e.keyCode === 229) return;

    // ? 鍵（Shift + /）：開關快捷鍵指南
    if (e.code === 'Slash' && e.shiftKey) {
      e.preventDefault();
      view.toggleModal();
      return;
    }

    // Esc 鍵：若指南開啟則關閉
    if (e.code === 'Escape') {
      const el = $('#modal-kb');
      if (el && el.classList.contains('open')) {
        e.preventDefault();
        view.toggleModal(false);
        return;
      }
    }

    // 若 Modal 處於開啟狀態，暫停其他操作以避免干擾
    const modalOpen = $('#modal-kb') && $('#modal-kb').classList.contains('open');
    if (modalOpen) return;

    // 單鍵操作（Q, W, [, ] 等）必須確保沒有按下 Ctrl/Meta/Alt，避免搶佔瀏覽器或系統快捷鍵
    const noModifiers = !e.ctrlKey && !e.metaKey && !e.altKey;

    // [ 鍵：切換左側欄收合
    if (noModifiers && e.code === 'BracketLeft') {
      e.preventDefault();
      view.toggleL();
      return;
    }

    // ] 鍵：切換右側欄收合
    if (noModifiers && e.code === 'BracketRight') {
      e.preventDefault();
      view.toggleR();
      return;
    }

    // Q 鍵：Q head 循環 -1
    if (noModifiers && e.code === 'KeyQ') {
      e.preventDefault();
      state.sel.h = (state.sel.h - 1 + state.cfg.Hq) % state.cfg.Hq;
      state.sel.mi = null;
      view.render();
      return;
    }

    // W 鍵：Q head 循環 +1
    if (noModifiers && e.code === 'KeyW') {
      e.preventDefault();
      state.sel.h = (state.sel.h + 1) % state.cfg.Hq;
      state.sel.mi = null;
      view.render();
      return;
    }

    // 左右方向鍵：上一步 / 下一步（支援 Ctrl / Cmd 極值跳轉）
    if (e.code === 'ArrowRight') {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) state.go(state.steps.length - 1);
      else state.go(state.sel.step + 1);
      return;
    }
    if (e.code === 'ArrowLeft') {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) state.go(0);
      else state.go(state.sel.step - 1);
      return;
    }

    // 上下方向鍵：跨 Layer 切換同算子階段
    if (e.code === 'ArrowUp' || e.code === 'ArrowDown') {
      e.preventDefault();
      const cur = state.steps[state.sel.step];
      if (!cur || cur.layer < 0 || cur.layer >= state.cfg.L) return;

      const targetLayer = e.code === 'ArrowUp' ? cur.layer - 1 : cur.layer + 1;
      if (targetLayer < 0 || targetLayer >= state.cfg.L) return;

      const targetIdx = state.steps.findIndex(s => s.layer === targetLayer && s.name === cur.name);
      if (targetIdx !== -1) state.go(targetIdx);
      return;
    }
  });

  // 初始啟動
  state.loadModel('baseline', true);

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
