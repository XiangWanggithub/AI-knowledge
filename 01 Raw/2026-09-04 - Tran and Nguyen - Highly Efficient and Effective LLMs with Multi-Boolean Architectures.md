# Highly Efficient and Effective LLMs with Multi-Boolean Architectures (MBOK)

- Canonical URL: https://arxiv.org/abs/2505.22811
- PDF URL: https://arxiv.org/pdf/2505.22811
- OpenReview: https://openreview.net/forum?id=r0CH5dF3Se
- Source type: arXiv (URL-only, Tier 1)
- Accessed at: 2026-09-04 Asia/Shanghai
- arXiv ID: 2505.22811 (v1 2025-05-28 → v5 2026-04-21) · stat.ML, cs.LG
- Venue: **ICLR 2026 (Main Conference)** — arXiv comments field
- Authors: Ba-Hien Tran, Van Minh Nguyen — **Huawei Paris Research Center** (corresponding: ba.hien.tran@huawei.com)
- Open source: **none located** (checked 2026-09-04): no code link on arXiv or OpenReview; the paper's only repo citations are third-party (BitBLAS, QuIP#, QTIP); web search found no release
- Tier: 1
- Raw-artifact decision: PDF 1.57 MB (< 2 MB) but arXiv PDFs default to URL-only → **not committed**

## Raw capture
> Verification: method, Table 2, Table 4 and limitations **hand-verified against the full PDF (v5, 42 pp incl. App. A–G) on 2026-09-04** via pypdf text extraction. High confidence. Venue from the arXiv comments field; open-source status = absence of evidence (medium confidence).

### Abstract (verbatim)
Weight binarization has emerged as a promising strategy to reduce the complexity of large language models (LLMs). Existing approaches fall into post-training binarization, which is simple but causes severe performance loss, and training-aware methods, which depend on full-precision latent weights, adding complexity and limiting efficiency. We propose a novel framework that represents LLMs with multi-kernel Boolean parameters and, for the first time, enables direct finetuning LMMs in the Boolean domain, eliminating the need for latent weights. This enhances representational capacity and dramatically reduces complexity during both finetuning and inference. Extensive experiments across diverse LLMs show our method outperforms recent ultra low-bit quantization and binarization techniques.

*("LMMs" typo is in the original.)*

### Core thesis
Two moves. (1) **Multi-Boolean kernels**: approximate an FP weight matrix as a sum of K binary (±1) matrices, each with its own rank-1 FP scaling — a residual binary decomposition, so K kernels ≈ K bits/weight but with a non-uniform, position-dependent codebook. (2) **Native Boolean training**: optimize the ±1 weights *directly in the Boolean domain* by a flip rule driven by an accumulated Boolean loss signal — no FP16 latent weights, no STE — extending Nguyen et al. (2024) "BOLD" from small nets to LLMs. Result: ultra-low-bit (2-kernel = 2-bit) weights close to FP16 quality, far cheaper finetuning, and real measured GPU speedups.

### Method (verified, §3–5)
- **Pitfall of latent weights (§3.1):** `W_bin = sign(W_FP)`; training must keep the FP copy and update `W_bin = sign(W_FP − η·G)` through an STE proxy → with Adam, 3×16 bits of state per parameter; STE noise causes oscillation.
- **Native Boolean layer (§3.2, from BOLD):** forward `Y[k,j] = b[j] + Σ_i L(X[k,i], W[i,j])` with logic gate `L` (xnor used); mixed-type rule for real input `x` and Boolean weight `w`: `xnor(w,x) = s` with `s_bool = xnor(w_bool, x_bool)` and `|s| = |x|`. **Prop. A.10: `xnor(w,s) = w × s`**, so it maps onto ordinary linear algebra. Backward: Boolean loss signal `Q[i,j] = Σ_k 1(Q_kij=TRUE)|Q_kij| − Σ_k 1(Q_kij=FALSE)|Q_kij|`, `Q_kij = xnor(Z[k,j], X[k,i])`. **Boolean optimizer:** flip `W[i,j] ← ¬W[i,j]` if `xnor(Q[i,j], W[i,j]) = TRUE`; accumulator `M^{t+1} = β_t M^t + η Q^t` with `β_t` set adaptively via brain-plasticity / Hebbian heuristics (App. A). One FP momentum per parameter.
- **SVID (§4.1, from OneBit):** `W ≈ W_bool ⊙ (s_out s_inᵀ)`, `W_bool = sign(W)`, `s_in = √σ₁ V[:,1]`, `s_out = √σ₁ U[:,1]` from the rank-1 SVD of `|W|`. Prop. 4.1 (restated from OneBit): beats plain rank-1 of `W`. **Prop. 4.3 (new):** the SVID scaling is Frobenius-optimal among all rank-1 magnitude factors `c dᵀ` given `W_bool`. Layer: `XWᵀ ≈ ((X ⊙ s_inᵀ) W_bool) ⊙ s_outᵀ`.
- **Multiple Boolean kernels (§4.2):** `W ≈ Σ_{k=1}^{K} W_bool^[k] ⊙ (s_out^[k] s_in^[k]ᵀ)`; `XWᵀ ≈ Σ_k ((X ⊙ s_in^[k]ᵀ) W_bool^[k]) ⊙ s_out^[k]ᵀ` (Eq. 7). Per kernel: 1 bit/weight + (m+n) FP scalars → **K kernels ≈ K bits/weight**; tables write it `K×1`. Multiplications inside each Boolean GEMM become additions.
- **Successive extraction (§4.3.1):** data-free init; kernel k is the SVID of the previous residual `W_res^[k] = W_in^[k] − W_bool^[k] ⊙ (s_out^[k] s_in^[k]ᵀ)` (Eq. 8).
- **KD finetuning (§4.3.2):** FP teacher → Boolean student; `L = L_logits + γ L_is`, logits term = forward KL at τ=1 (Eq. 10), `L_is` = MSE over selected hidden states (Eq. 11), γ=10. **Only the last kernel + scaling factors are trained** (§6.1.2: lowest flip rate, best ppl; lower-order kernels hold the bulk of the signal and perturbing them cascades into higher-order residuals).
- **Kernel allocation (§5):** choose `K_l` per weight matrix to minimize `E(k) = Σ_l h_l · e_l^{[K_l]} · f(p_l)`, `f(p) = (1/p) log(1/p)`, subject to expansion ratio `ρ(k) = Σ_l K_l p_l ≤ T`, `K_l ≤ K_max`. `e` = successive-SVID residual norm; `h_l` = PWCCA importance (App. E.1); `p_l` = relative size. NP-hard → greedy: repeatedly increment the `K_l` giving the largest `E` drop (Alg. 9). **Supports fractional average bit-widths.**

### Setup & results (verified, §6)
- Protocol of MoS (Jo et al. 2024); **activations are not quantized**. Train on WikiText2 + C4 subset, seq 2048, 3 epochs, batch 8; Boolean lr 5e-3, remaining FP params AdamW 2e-5.
- **Table 2 (K=2 → "2×1" bits; ppl Wiki2 / C4; zero-shot avg of BoolQ/PIQA/Hella/WinoG/ARC-e/ARC-c):**
  - OPT-1.3B — FP16 14.62/14.72 (53.99) · **MBOK 16.13/16.61 (51.69)** · MoS(1b) 18.45/18.83 · OneBit(1b) 20.36/20.76 · OmniQuant(2b) 42.43/55.64 · BiLLM 69.45 · OPTQ(2b) 9.5e3 · LLM-QAT(2b) 4.9e3
  - LLaMA-7B — FP16 5.68/7.08 (64.06) · **MBOK 6.83/8.53 (58.76)** · MoS 7.97/9.72 (54.48) · OneBit 8.48/10.49 · OmniQuant(2b) 15.34/26.21 (48.17) · OPTQ(2b) 1.9e3
  - LLaMA-13B — FP16 5.09/6.61 (66.39) · **MBOK 6.17/7.88 (61.45)** · MoS 7.16/8.81 · OneBit 7.65/9.56 · OmniQuant(2b) 13.43/19.33 · BiLLM(1.11b) 14.56/16.67
- **§6.3 / Fig. 1 / Table 3:** with 3 kernels MBOK closely approaches FP on OPT-125M…6.7B and Pareto-dominates RTN-3bit and OPTQ-3bit at equal model size.
- **§6.1:** deviation and finetuned ppl fall monotonically with #kernels (1→8); finetuning only the last kernel beats finetuning the first or all.
- **§6.6 / Fig. 11:** finetuning memory (OPT-6.7B, 3 kernels) vs MoS — 1 bit/weight + one 16-bit momentum vs 16-bit latent + two Adam momenta.
- **Table 4 — measured latency (A100, batch 1, BitBLAS INT1-weight × FP16-activation kernels), LLaMA-13B linear layers, ms (speedup vs FP16):** 5120×5120 0.1654 → **0.0507 (3.25×)**; 5120×13824 0.4283 → **0.0510 (8.40×)**; 13824×5120 0.4341 → **0.0499 (8.70×)**. Same table: QuIP# 0.27–0.69× and QTIP 0.08–0.09× — the 2-bit **vector-quantization baselines run slower than FP16** because of dequant overhead (App. G.11).
- LLaMA2-7B/13B and other K in App. G.3–G.4; more baselines App. G.6; VQ discussion App. G.11.

### Stated limitations (§7 + observed)
- No native Boolean accelerator exists; all measurements are on real-arithmetic GPUs (INT1 path via BitBLAS) — the full "native Boolean" benefit is deferred to future hardware.
- Weight-only: activations stay FP16, so not comparable to W4A4-class methods.
- (observed) "No STE / no latent weights" holds for the Boolean kernels; scaling vectors and other FP parameters are still trained with AdamW.
- (observed) No code released.

### Key related work cited
- Parent framework: Nguyen et al. 2024 — BOLD, Boolean Logic Deep Learning (native Boolean training). Baselines: OneBit, MoS (QAT with latent weights); BiLLM, PB-LLM, STBLLM, ARB-LLM (hybrid PTQ, salient weights higher-bit); BitStack, QBB, DB-LLM (multiple binary bases); BitNet (train from scratch); OPTQ, OmniQuant, LLM-QAT (2-bit); QuIP#, QTIP (vector quantization). In-the-wild follow-up: RaBiT (arXiv:2602.05367, residual-aware binarization training).
