# Toy Llama Block Stepper

**English** | [繁體中文](./README.zh-Hant.md)

A lightweight, pure-frontend **step-by-step calculation visualizer and interactive lab for LLM Transformer Blocks**.

This project breaks down a standard Llama/Mistral architecture (including GQA, RoPE, SwiGLU MLP, and RMSNorm) into compact, human-scale dimensions so you can **step through operations manually**, **click any cell to inspect vector dot-product arithmetic**, and **live-edit the input matrix $X_0$** to observe how values flow across layers.

> 📐 **Design Philosophy & Architectural Invariants**: See [`docs/DESIGN.md`](./docs/DESIGN.md) ([繁體中文](./docs/DESIGN.zh-Hant.md)) for the project's core principles, non-negotiable constraints, and non-goals.

---

## 🌟 Features

* **No Build Step, Runs Anywhere**: Vanilla HTML/CSS/JavaScript + MathJax. No `npm install` needed—open directly in a browser or deploy straight to GitHub Pages.
* **Mechanism Demo Presets**: Includes a "Random Baseline" model alongside models with hand-crafted mechanisms such as a "Previous-Token Head" (demonstrating an attention head specialized in attending to $t-1$) and a "SwiGLU Feature Gate".
* **Full Arithmetic Drill-Down (Micro View)**: Click any cell in an output matrix to expand its exact term-by-term multiply-accumulate or softmax normalization steps.
* **Model–Input Separation**: Model architectures and weights are loaded as independent static assets. Freely adjust `seq_len`, `seed`, or directly edit cells in $X_0$ with instant forward recomputation.
* **Dynamic Residual Norm Chart**: Real-time SVG line chart tracking the residual stream L2 norm $\Vert{}x_t\Vert{}$ at each stage.
* **Light & Dark Themes**: Defaults to light mode with a one-click `🌓` toggle for dark mode.
* **Three-Axis Keyboard Navigation**: Navigate across steps (`←`/`→`), jump to endpoints (`Ctrl+←`/`→`), compare the same stage across layers (`↑`/`↓`), and cycle query heads (`Q`/`W`), with a built-in shortcut guide modal (`?`).
* **Collapsible Sidebars**: Toggle the left step list (`[` or `⇤`) and right inspector panel (`]` or `⇥`) at any time to maximize viewport space.

---

## 📐 Transformer Block Architecture

Each layer’s Transformer block follows this flow:

```mermaid
flowchart TD
    X[Input Residual Stream X]

    subgraph AttnBlock [1. Attention Module: GQA + RoPE]
        N1[Attn Pre-RMSNorm]
        QKV[Q / K / V Linear Projections]
        ROPE[Apply RoPE to Q & K]
        SCORE[Scores + Causal Mask + Softmax]
        AV[Weighted Sum A · V]
        O[o_proj Output Projection]

        N1 --> QKV
        QKV --> ROPE
        ROPE --> SCORE
        SCORE --> AV
        AV --> O
    end

    ADD1((+ Residual Add))

    subgraph MLPBlock [2. Feed-Forward Module: SwiGLU MLP]
        N2[MLP Pre-RMSNorm]
        GU[Gate & Up Projections]
        SWI[SwiGLU Activation: SiLU(Gate) ⊙ Up]
        DOWN[down_proj Back to d_model]

        N2 --> GU
        GU --> SWI
        SWI --> DOWN
    end

    ADD2((+ Residual Add))
    OUT[Output to Next Layer Residual Stream]

    X --> N1
    X --> ADD1
    O --> ADD1

    ADD1 --> N2
    ADD1 --> ADD2
    DOWN --> ADD2

    ADD2 --> OUT
```

---

## ⚙️ Model Specifications & Input Settings

### 1. Model Specifications (Read-Only, Defined by Preset)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `d_model` | `16` | Hidden dimension (fixed) |
| `L` | `4` | Number of Transformer layers (fixed) |
| `Hq` | `4` | Number of Query heads |
| `Hkv` | `2` | Number of Key/Value heads (GQA sharing ratio `Hq / Hkv`) |
| `hd` | `4` | Head dimension |
| `ffm` | `3` | MLP hidden dimension multiplier (`d_model * ffm`) |
| `theta` | `10000` | RoPE base $\Theta$ |

### 2. Input & Inference Settings (Interactive in Right Sidebar)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `T` (`seq_len`) | `5` | Sequence length in tokens (range: 3–8) |
| `seed` | `7` | PRNG seed for generating the input matrix $X_0$ |

---

## 📁 Project Directory Structure

```text
toy-llama-stepper/
├── index.html               # Semantic HTML shell
├── README.md
├── README.zh-Hant.md
├── docs/
│   ├── DESIGN.md
│   └── DESIGN.zh-Hant.md
├── css/
│   └── style.css            # UI styles & theme variables
├── js/
│   ├── vendor/
│   │   └── tex-svg.min.js   # Third-party MathJax rendering engine
│   ├── state.js             # Centralized state management
│   ├── steps.js             # Step metadata & formula templates
│   ├── model.js             # Pure math & forward inference engine
│   ├── view.js              # DOM & SVG chart rendering
│   └── app.js               # Entry point & event listeners
├── models/
│   ├── models-static.js     # Prebuilt static model weights
│   └── presets.js           # Dynamic model generator script
├── resources/
│   ├── icon-dark.svg
│   ├── icon-light.svg
│   └── og-preview.png       # Open Graph link preview image
└── scripts/
    └── build-static.js      # Static weight build utility
```

---

## 🚀 Quick Start & Deployment

### 1. Run Locally
No environment setup required—clone the repository and open `index.html` directly in your browser:
```bash
git clone https://github.com/JunliXiao/toy-llama-stepper.git
# Open index.html directly in your browser
```
*(Node.js is only required if you want to regenerate static weight files via `scripts/build-static.js`.)*

### 2. Deploy to GitHub Pages
1. Push the repository to GitHub.
2. Go to **Settings** -> **Pages**.
3. Select the `main` branch and root directory (`/root`), then save to publish.