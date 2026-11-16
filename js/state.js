/**
 * Toy Llama Stepper - 集中狀態管理模組 (Centralized State Management)
 */
(function (global) {
  'use strict';

  global.ToyLlama = global.ToyLlama || {};

  const detectLang = () => {
    const nav = typeof navigator !== 'undefined' ? navigator.language || '' : '';
    return /^zh/i.test(nav) ? 'zh-Hant' : 'en';
  };

  const state = {
    cfg: {
      T: 5,
      seed: 7,
      d: 16,
      L: 4,
      Hq: 4,
      Hkv: 2,
      hd: 4,
      ffm: 3,
      theta: 10000
    },
    W: null,
    X0: null,
    steps: [],
    ck: [],
    sel: {
      step: 0,
      h: 0,
      mi: null,
      model: 'baseline',
      lang: detectLang()
    },
    pendingFocus: null,

    initInput() {
      const { rng, rmat } = global.ToyLlama.model;
      const r = rng(state.cfg.seed + 999);
      state.X0 = rmat(r, state.cfg.T, state.cfg.d, 1);
    },

    recompute(resetStep = false, renderOpts = {}) {
      if (resetStep) state.sel.step = 0;
      if (!renderOpts.keepMi) state.sel.mi = null;
      global.ToyLlama.model.forward(state);
      if (state.sel.step >= state.steps.length) {
        state.sel.step = Math.max(0, state.steps.length - 1);
      }
      if (global.ToyLlama.view) {
        global.ToyLlama.view.render(renderOpts);
      }
    },

    go(idx) {
      if (!state.steps.length) return;
      const next = Math.max(0, Math.min(state.steps.length - 1, idx));
      state.sel.step = next;
      state.sel.mi = null;
      if (global.ToyLlama.view) {
        global.ToyLlama.view.render();
      }
    },

    loadModel(modelId, isInit = false) {
      const models = global.TOY_MODELS || {};
      const keys = Object.keys(models);
      const targetKey = models[modelId] ? modelId : keys[0];
      const m = models[targetKey];
      if (!m) return;

      state.sel.model = targetKey;
      state.W = m;
      Object.assign(state.cfg, m.config);
      if (state.sel.h >= state.cfg.Hq) state.sel.h = 0;

      const view = global.ToyLlama.view;
      if (isInit) {
        state.initInput();
        if (view) {
          view.applyStaticI18n();
          view.fillMs();
          view.fillHs();
        }
      } else if (view) {
        view.fillHs();
      }

      state.recompute(isInit);
    },

    setLang(nextLang) {
      if (state.sel.lang === nextLang) return;
      state.sel.lang = nextLang;
      const view = global.ToyLlama.view;
      if (view) {
        view.applyStaticI18n();
        view.fillMs();
      }
      state.recompute(false, { keepMath: true, keepMi: true });
    }
  };

  global.ToyLlama.state = state;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);