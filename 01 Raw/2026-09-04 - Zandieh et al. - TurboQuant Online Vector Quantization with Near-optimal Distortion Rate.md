# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

- Canonical URL: https://arxiv.org/abs/2504.19874
- PDF URL: https://arxiv.org/pdf/2504.19874
- Source type: arXiv (URL-only, Tier 1)
- Accessed at: 2026-09-04 Asia/Shanghai
- arXiv ID: 2504.19874 (v1 2025-04-28; the only version) · cs.LG, cs.AI, cs.DB, cs.DS
- Venue: **ICLR 2026** — per multiple independent third-party sources (Wikipedia entry; several reimplementations titled "ICLR 2026"); **not stated on the arXiv page** (no comments field) → medium confidence
- Authors: Amir Zandieh (Google Research), Majid Daliri (New York University), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)
- Open source: **no official release found** (2026-09-04). Third-party reimplementations: github.com/OmarHory/turboquant, github.com/OnlyTerp/turboquant, github.com/scos-lab/turboquant, github.com/vivekvar-dl/turboquant (`pip install turbokv`), github.com/Pascal-SAPUI5/llama.cpp-turboquant (ROCm); PyPI `turboquant`. Their speed claims (e.g. "1.85× quantized attention at 16K") are **unverified by the paper**.
- Follow-ups in the wild: Fast-TurboQuant (arXiv:2606.21448, multiplier-free); "Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant" (arXiv:2605.08114)
- Tier: 1
- Raw-artifact decision: PDF 0.86 MB; arXiv → **URL-only, not committed**

## Raw capture
> Verification: problem definition, Algorithms 1–2, Theorems 1–3 statements, and all experiment tables **hand-verified against the full PDF (v1, 25 pp) on 2026-09-04** via pypdf. High confidence. Venue: third-party only (medium).

### Abstract (verbatim)
Vector quantization, a problem rooted in Shannon's source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. We propose TurboQuant to address both mean-squared error (MSE) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates. Our data-oblivious algorithms, suitable for online applications, achieve near-optimal distortion rates (within a small constant factor) across all bit-widths and dimensions. TurboQuant achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates, and leveraging the near-independence property of distinct coordinates in high dimensions to simply apply optimal scalar quantizers per each coordinate. Recognizing that MSE-optimal quantizers introduce bias in inner product estimation, we propose a two-stage approach: applying an MSE quantizer followed by a 1-bit Quantized JL (QJL) transform on the residual, resulting in an unbiased inner product quantizer. We also provide a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer, demonstrating that TurboQuant closely matches these bounds, differing only by a small constant (≈ 2.7) factor. Experimental results validate our theoretical findings, showing that for KV cache quantization, we achieve absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel. Furthermore, in nearest neighbor search tasks, our method outperforms existing product quantization techniques in recall while reducing indexing time to virtually zero.

### Problem (§1.1)
Design `Q: ℝᵈ → {0,1}^{b·d}` (bit-width `b` per coordinate) with dequantizer `Q⁻¹`, **worst-case inputs**, randomized `Q`, minimizing expected **MSE** `D_mse = E‖x − Q⁻¹(Q(x))‖²` and **inner-product error** `D_prod = E(⟨y,x⟩ − ⟨y,Q⁻¹(Q(x))⟩)²`, with the additional requirement that the inner-product quantizer be **unbiased**. Two primitives: `Quant` (dataset) and `DeQuant` (any item). No data assumptions → online / data-oblivious.

### Method (§1.3, §2.2, §3; verified)
- **Key fact (Lemma 1):** for `x` uniform on the unit sphere `S^{d−1}`, each coordinate follows a scaled Beta density `∝ (1 − x²)^{(d−3)/2}` → `N(0, 1/d)` as `d` grows; distinct coordinates are nearly independent in high `d`. A **random rotation Π** therefore turns *any* input into this known distribution regardless of the data.
- **TurboQuant_mse (Alg. 1):** rotate `Π·x`; quantize **each coordinate independently** with the **optimal Lloyd-Max scalar quantizer** for the Beta density (continuous 1-D k-means); codebooks are **precomputed once per bit-width**; dequantize and rotate back. Unit-norm assumption is standard; store ‖x‖ in FP otherwise.
- **QJL (Def. 1, from Zandieh et al. [62]):** `Q_qjl(x) = sign(S x)`, `S ∈ ℝ^{d×d}` i.i.d. `N(0,1)`; dequant `√(π/2)/d · Sᵀ z`. Lemma 4: **unbiased** inner products, `Var ≤ (π/2d)‖y‖²`.
- **TurboQuant_prod (Alg. 2):** run TurboQuant_mse at **b−1 bits**, then apply **1-bit QJL to the residual** `r`; output `(idx, qjl bits, ‖r‖₂)`; dequant = `x̃_mse + x̃_qjl`. Motivation: MSE-optimal quantizers are **biased** for inner products at low bit-widths (bias grows with the average inner product, §4.1 Fig. 2); the QJL stage removes the bias.
- **Fractional bit-widths in practice (§4.3):** split channels into outlier / non-outlier sets and run two TurboQuant instances at different bit-widths — e.g. 2.5-bit = 32 outlier channels at 3 bits + 96 at 2 bits of 128; 3.5-bit uses a different ratio.

### Theory (verified statements)
- **Theorem 1 (MSE):** for `‖x‖=1`, `D_mse ≤ (√3π/2)·4^{−b}` for all `b ≥ 0`; refined `b=1,2,3,4 → ≈ 0.36, 0.117, 0.03, 0.009`.
- **Theorem 2 (inner product):** unbiased; `D_prod ≤ (√3π²‖y‖²/d)·4^{−b}`; refined `b=1..4 → ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d`.
- **Theorem 3 (lower bound, Shannon + Yao minimax):** any randomized `b`-bit quantizer has hard instances with `D_mse ≥ 4^{−b}` and `D_prod ≥ (‖y‖²/d)·4^{−b}`.
- ⇒ TurboQuant is within **√3π/2 ≈ 2.7×** of the information-theoretic optimum for all `b`, and within **≈1.45×** at `b=1`; exponential improvement in bit-width dependence over prior methods.

### Experiments (§4; single NVIDIA A100)
- **§4.1 validation:** DBpedia entities, OpenAI text-embedding-3 (1536-d), 100k train / 1k queries. TurboQuant_prod unbiased at all `b`; TurboQuant_mse biased, converging as `b` grows (Figs. 1–2); empirical MSE and inner-product errors sit between the proven upper and lower bounds (Fig. 3).
- **§4.2 Needle-in-a-haystack (Fig. 4):** Llama-3.1-8B-Instruct, 4k–104k tokens, every method at **KV memory ratio 0.25**. Scores — SnapKV 0.858, PyramidKV 0.895, KIVI 0.981, PolarQuant 0.995, **Full-precision 0.997, TurboQuant 0.997** (identical to FP at >4× compression). Theory-backed quantizers (PolarQuant, TurboQuant) beat token-eviction (SnapKV, PyramidKV) and plain scalar quant (KIVI).
- **§4.3 LongBench-E, Table 1 (avg over SingleQA/MultiQA/Summ/Few-shot/Synthetic/Code):** Llama-3.1-8B-Instruct — Full cache (16-bit) **50.06**; KIVI 3-bit 48.50; KIVI 5-bit 50.16; PolarQuant 3.9-bit 49.78; **TurboQuant 2.5-bit 49.44**; **TurboQuant 3.5-bit 50.06** (= full). Ministral-7B-Instruct — Full 49.89; **TurboQuant 2.5-bit 49.62**. TurboQuant quantizes **generated tokens too** (KIVI/PolarQuant leave them unquantized); compression ≥ 4.5×.
- **§4.4 Near-neighbour search:** DBpedia OpenAI3 1536-d and 3072-d, GloVe 200-d; recall `1@k` vs Product Quantization (k-means codebooks, LUT256, trained on the same data) and RabitQ — TurboQuant highest recall in every setting at 2 and 4 bits. **Table 2 quantization/indexing time (4-bit):** PQ 37.04 / 239.75 / 494.42 s, RabitQ 597.25 / 2267.59 / 3957.19 s, **TurboQuant 0.0007 / 0.0013 / 0.0021 s** (d = 200 / 1536 / 3072).

### Limitations (observed — v1 has no dedicated limitations section)
- **No wall-clock attention latency or decode throughput is reported**; efficiency claims in the paper are compression ratio, indexing time and vectorizability. Speedups circulating online come from third-party reimplementations.
- KV-cache evaluation limited to two 7–8B instruct models (Llama-3.1-8B, Ministral-7B) and LongBench/needle; no very-large models, no vision/VLA contexts.
- Dense `d×d` rotation and QJL matrices are assumed; structured (e.g. Hadamard) variants are natural but not the paper's subject.
- Unit-norm theory; norms stored separately in FP.
- Which quantizer (mse vs prod) is applied to keys vs values in the KV experiments is not spelled out in the v1 text read here.

### Key related work cited
- Theory lineage: Shannon source coding; Zador; Gersho (lattice VQ). Online KV-cache quant: KIVI, KVQuant, PolarQuant, QJL (the authors' own 1-bit sketch); token eviction: SnapKV, PyramidKV. Offline weight PTQ using Hessians (OPTQ/GPTQ family) as the contrasting *data-dependent* camp. NN-search: Product Quantization, RabitQ; vector DBs (pgvector etc.).
