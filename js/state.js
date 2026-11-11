/**
 * Toy Llama Stepper - 集中狀態管理 (State Management)
 */
(function (global) {
  'use strict';

  global.ToyLlama = global.ToyLlama || {};

  const defaultCfg = {
    d: 16,
    L: 4,
    Hq: 4,
    Hkv: 2,
    hd: 4,
    ffm: 3,
    theta: 10000,
    T: 5,
    seed: 7
  };

  const state = {
    cfg: { ...defaultCfg },
    sel: { step: 0, h: 0, mi: null, model: 'baseline' },
    W: null,
    X0: [],
    steps: [],
    ck: [],
    pendingFocus: null,

    initInput(regen) {
      if (regen || !state.X0 || state.X0.length !== state.cfg.T) {
        const M = global.ToyLlama.model;
        state.X0 = M.rmat(M.rng(state.cfg.seed + 999), state.cfg.T, state.cfg.d, 1);
      }
    },

    makeWeights(regenX) {
      const { d, L, Hq, Hkv, hd } = state.cfg;
      const ff = d * state.cfg.ffm;
      const M = global.ToyLlama.model;
      const r = M.rng(state.cfg.seed);

      state.W = { layers: [], gf: M.ones(d) };
      for (let l = 0; l < L; l++) {
        state.W.layers.push({
          g1: M.ones(d),
          g2: M.ones(d),
          Wq: M.rmat(r, d, Hq * hd, 1 / Math.sqrt(d)),
          Wk: M.rmat(r, d, Hkv * hd, 1 / Math.sqrt(d)),
          Wv: M.rmat(r, d, Hkv * hd, 1 / Math.sqrt(d)),
          Wo: M.rmat(r, Hq * hd, d, 1 / Math.sqrt(Hq * hd)),
          Wg: M.rmat(r, d, ff, 1 / Math.sqrt(d)),
          Wu: M.rmat(r, d, ff, 1 / Math.sqrt(d)),
          Wd: M.rmat(r, ff, d, 1 / Math.sqrt(ff))
        });
      }
      if (regenX) state.initInput(true);
    },

    loadModel(modelId, regenX = false) {
      state.sel.model = modelId;
      const models = global.TOY_MODELS || {};
      const m = models[modelId] || models['baseline'];
      const M = global.ToyLlama.model;

      if (m) {
        Object.assign(state.cfg, m.config);
        state.W = {
          layers: m.layers,
          gf: m.gf || M.ones(state.cfg.d)
        };
      } else {
        state.makeWeights(regenX);
      }
      state.initInput(regenX);
      state.sel.h = Math.min(state.sel.h, state.cfg.Hq - 1);

      if (global.ToyLlama.view) {
        global.ToyLlama.view.fillHs();
        global.ToyLlama.view.fillMs();
      }
      if (global.ToyLlama.model) {
        global.ToyLlama.model.forward();
      }
      state.sel.mi = null;
      if (global.ToyLlama.view) {
        global.ToyLlama.view.render();
      }
    },

    recompute(regenX) {
      if (regenX) state.initInput(true);
      if (global.ToyLlama.model) {
        global.ToyLlama.model.forward();
      }
      state.sel.mi = null;
      if (global.ToyLlama.view) {
        global.ToyLlama.view.render();
      }
    },

    go(i) {
      state.sel.step = Math.max(0, Math.min(state.steps.length - 1, i));
      state.sel.mi = null;
      if (global.ToyLlama.view) {
        global.ToyLlama.view.render();
      }
    }
  };

  global.ToyLlama.state = state;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
