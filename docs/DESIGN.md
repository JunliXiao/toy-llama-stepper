# Toy Llama Block Stepper — Design Philosophy & Architectural Invariants

**English** | [繁體中文](./DESIGN.zh-Hant.md)  
← Back to [README](../README.md)

This document defines the core goals, architectural invariants, and design boundaries of **Toy Llama Block Stepper**. All future feature additions, UI changes, and model presets must adhere to these principles.

---

## 1. Design Philosophy & Goals

* **Demystify Mechanistic Interpretability**: Help learners without a strong math background build intuition for how LLMs extract features and route information, by watching actual numbers flow through a small model.
* **Fast and Intuitive Keyboard Navigation**: Design exploration around snappy, intuitive keyboard controls so users naturally develop muscle memory for navigating a Transformer's structure (steps, layers, and attention heads).
* **Transparent, Human-Scale Computation**: Reject black-box abstractions. Keep tensor dimensions small enough to fit comfortably within a single screen and human working memory, showing every matrix operation explicitly, with drill-down to elementary arithmetic.
* **Zero-Build, Run-Anywhere Portability**: Use only standard web technologies—no build step, bundler, or backend—so the app runs immediately whether opened directly from the local filesystem or hosted on static platforms like GitHub Pages.
* **Stateless Sandbox**: Treat the tool as an "interactive scratchpad for calculations" that persists no user state or history, allowing learners to freely tweak inputs and experiment without fear of breaking anything—a simple page refresh always restores a clean slate.

---

## 2. Architectural Invariants

* **Zero-Build & `file://` Protocol Compatibility**: The codebase must remain pure static vanilla HTML/CSS/JS without bundlers or frameworks. All application scripts, math rendering engine, and default model weights must load via standard static `<script>` tags, must avoid anything that triggers CORS restrictions under `file://` (such as native ES Modules or runtime `fetch()` calls for local assets).
* **Stateless & Deterministic Execution**: The application must never persist user input edits, active step positions, or UI preferences to browser storage (`localStorage`, `IndexedDB`, or cookies). Given the same model weights and input parameters (`X₀`, `seq_len`, `seed`), opening the tool at any time or on any machine must produce identical numerical outputs and visual states.
* **Transparent, Unfused Computation & Micro-Level Drill-Down**: The forward pass must never fuse operators for performance. Every mathematical transformation must preserve its intermediate tensor as a separate step, and every cell in an output matrix must be inspectable down to elementary arithmetic (term-by-term dot products or softmax normalization).
* **Single-Screen Visibility & Human-Scale Dimensions**: Hidden dimensions, head counts, and sequence lengths must remain compact enough to be scanned in full on a single screen without virtual scrolling or truncated matrices, so value distributions and signs are visible at a glance.
* **Model–Input Separation**: Preloaded model architectures and weights must remain strictly decoupled from inference inputs. Editing `X₀` or changing the random seed in the UI may only trigger a pure, deterministic recomputation of the forward pass; it must never modify the loaded model weights.
* **Cross-Layer Structural Symmetry**: The order, number, and naming of steps within every Transformer block must remain strictly symmetrical across layers, guaranteeing that three-axis navigation (intra-layer stepping, cross-layer stage comparison, and head cycling) stays predictable.
* **Input Precedence & Non-Intrusive Shortcuts**: Global keyboard shortcuts (`←`/`→`, `↑`/`↓`, `Q`/`W`, `[`/`]`) must yield to native text editing and OS/browser controls under three boundary conditions:
  1. **Form Focus Yielding**: Global shortcuts are completely disabled whenever focus is inside any editable input element (such as `X₀` matrix cells or parameter fields).
  2. **IME Composition Safety**: No key events may be intercepted while an Input Method Editor (IME) is actively composing text (`isComposing`), preventing interference with CJK character selection.
  3. **Physical Key Binding & Modifier Safety**: Single-key shortcuts must bind to physical key locations (`e.code`) rather than character outputs (`e.key`) and verify the absence of modifier keys (`Ctrl`, `Meta`, `Alt`). This ensures reliable behavior across multi-language keyboard layouts on macOS while never hijacking native browser/OS shortcuts (e.g., `Cmd+W`, `Cmd+R`, `Ctrl+Tab`).
* **Text–Engine Separation (i18n-Ready)**: Step titles, mathematical descriptions, and UI strings must remain cleanly separated from the core numerical engine (`model.js`), allowing new languages to be added without duplicating or altering math logic.

---

## 3. Non-Goals

* **No Backpropagation or Training Loops**: The tool focuses strictly on inference-time forward passes and inspecting known mechanisms; it will not include loss functions, gradient computation, backpropagation, or weight updates.
* **No Build Pipelines or Frontend Frameworks**: We reject package managers, transpilers, and component frameworks to preserve long-term readability, minimal maintenance, and instant local execution.
* **No User State Persistence**: There will be no user accounts, saved sessions, or restoring state from browser storage; reloading the page unconditionally resets the workspace.
* **No Large-Scale Weights or Real-Vocabulary Tokenization**: We will not load production-scale LLM weights or expand tensor dimensions to sizes requiring hardware acceleration or pagination, since that would undermine white-box transparency.
* **No Browser Shortcut Hijacking or Custom Keymaps**: We will not override browser or OS shortcuts, or introduce complex keymap configuration panels, keeping the interaction model lightweight and zero-config.