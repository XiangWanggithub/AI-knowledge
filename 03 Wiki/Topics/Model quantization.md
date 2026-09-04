# Model quantization

## Purpose
This topic page collects the vault's current and future work on model quantization, low-bit numerical representations, and related training/inference tradeoffs.

## Scope
This topic includes:
- post-training quantization (PTQ)
- quantization-aware training (QAT)
- low-bit training
- FP8 and low-bit floating-point formats
- scaling and calibration strategies
- precision/range tradeoffs in numeric representation design
- rotation-based outlier suppression (QuaRot / DuQuant / SVD·Hadamard)
- dynamic / input-conditioned mixed precision (runtime-adaptive bit allocation)
- domain-specific quantization (e.g. embodied / VLA models — see [[VLA quantization]])
- ultra-low-bit / binarization (≤2-bit; Boolean-kernel decomposition, native Boolean training)
- KV-cache / vector quantization (quantizing activations *at rest* rather than weights; online, data-oblivious)

## Current entry points
### Sources
- [[Luo et al. - Ascend HiFloat8 Format for Deep Learning]]
- [[Xiao et al. - SmoothQuant Accurate and Efficient Post-Training Quantization for Large Language Models]]
- [[Lin et al. - DuQuant Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs]]
- [[Zheng et al. - DyQ-VLA Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models]]
- [[Zhang et al. - QuantVLA Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models]]
- [[Wang et al. - Omega-QVLA Robust Quantization for Vision-Language-Action Models via Composite Rotation and Per-step Scaling]]
- [[Xu et al. - QVLA Not All Channels Are Equal in Vision-Language-Action Models Quantization]]
- [[Tran and Nguyen - Highly Efficient and Effective LLMs with Multi-Boolean Architectures]]
- [[Zandieh et al. - TurboQuant Online Vector Quantization with Near-optimal Distortion Rate]]

### Concepts / sub-clusters
- [[VLA quantization]] — VLA-specific low-bit quantization (application sub-cluster cutting across the routes below)

### Figures
- [[hif8_value_density.html|HiF8 ↔ FP8 representable-value density]] — interactive: values per octave on a log₂ axis (route 1, format design), making HiF8's tapered precision visible against flat E4M3 / E5M2

## Current thesis direction
This topic is still at an early stage in the vault, but a clearer internal structure is beginning to emerge. Quantization should not be treated only as a compression or deployment trick. In some settings, especially LLM quantization and low-bit training, the problem becomes one of choosing where to intervene in the numerical system.

Four routes are now visible:
1. **Representation design** — redesign the number format itself to improve the precision/range tradeoff (e.g. [[Luo et al. - Ascend HiFloat8 Format for Deep Learning]])
2. **Distribution reshaping / difficulty migration** — transform activations/weights so that standard low-bit quantization becomes viable and hardware-friendly. From [[Xiao et al. - SmoothQuant Accurate and Efficient Post-Training Quantization for Large Language Models]] (per-channel smoothing) the line extends into **rotation-based** outlier suppression (QuaRot, [[Lin et al. - DuQuant Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs|DuQuant]] — DuQuant first localized the "massive outliers" at the FFN down-projection and pairs a greedy data-aware block rotation with a zigzag permutation), reaching the two DuQuant-based VLA methods: [[Wang et al. - Omega-QVLA Robust Quantization for Vision-Language-Action Models via Composite Rotation and Per-step Scaling]] composes an SVD·Hadamard rotation with per-step scaling to push *uniform* W4A4 onto VLA diffusion action heads, while [[Zhang et al. - QuantVLA Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models]] instead keeps DiT attention FP16 (selective W4A8) plus two interface calibrations, prioritizing real integer-GEMM deployment.
3. **Dynamic / runtime-adaptive precision** — keep the format fixed but make the bit allocation a *function of the input or task state at inference*, spending precision only where the task is currently hard (e.g. [[Zheng et al. - DyQ-VLA Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models]]). Its **static sibling**, [[Xu et al. - QVLA Not All Channels Are Equal in Vision-Language-Action Models Quantization|QVLA]], allocates per-channel bits *offline* by action-space sensitivity (unifying quantization with 0-bit pruning) — the static form of the same importance-driven mixed-precision idea (HAWQ lineage, made action-guided)
4. **Boolean-kernel decomposition + native Boolean training** — change what a weight *is* and how it is *optimized*, rather than how an FP weight is rounded: [[Tran and Nguyen - Highly Efficient and Effective LLMs with Multi-Boolean Architectures|MBOK]] (Huawei Paris, ICLR 2026) represents each linear layer as a sum of K binary (±1) kernels with rank-1 FP scalings (K kernels ≈ K bits; successive SVID on residuals) and trains the bits **directly in the Boolean domain** with a flip rule and one momentum per parameter — no FP16 latent weights, no STE. 2-bit LLaMA-7B ppl 6.83 vs FP16 5.68 (OmniQuant-2bit 15.34), 3-bit ≈ FP; measured 8.7× per-layer speedup via BitBLAS INT1 kernels. This is the vault's first **≤2-bit / binarization** source and moves the topic's bit floor from ~4-bit to 2-bit (weight-only; activations FP16)

The first cut runs along **what you intervene on**: the *format* (route 1) vs. the *distribution* (route 2). Route 3 adds an orthogonal cut along **when the configuration is decided**: routes 1–2 both produce a *static* quantization config; route 3 makes precision a *runtime control variable*. Route 4 cuts deeper still — it changes the *parameterization and optimizer* rather than the rounding of an FP weight.

A **second axis — *what* is quantized** — became visible with the first KV-cache source. Routes 1–4 all quantize **weights** (and, for W4A4-class methods, activations in flight). [[Zandieh et al. - TurboQuant Online Vector Quantization with Near-optimal Distortion Rate|TurboQuant]] (Google Research/DeepMind + NYU, ICLR 2026) instead quantizes the **KV cache** — activations *at rest*, whose footprint scales with context length, not parameter count — with an **online, data-oblivious vector quantizer** that is provably within ≈2.7× of the Shannon distortion-rate bound: random rotation → Beta/Gaussian coordinates → precomputed optimal Lloyd-Max scalar codebook per coordinate, plus a 1-bit QJL residual sketch for **unbiased inner products**. 3.5 bits/channel is quality-neutral on LongBench (50.06 = full cache, Llama-3.1-8B), 2.5 bits costs −0.6. Two lessons for the routes: (i) it is the *distortion-rate theory* behind Route 2's rotate-then-scalar-quantize move — DuQuant/Ω-QVLA's data-aware rotations are refinements inside that guarantee; (ii) KV entries are produced during generation, so this axis forces the online/zero-calibration constraint that offline weight PTQ never faces. Keep the two cuts separate: *how* (routes 1–4) × *what* (weights / activations / KV cache).

A useful caution comes from the four VLA-quantization sources, which now form their own sub-cluster ([[VLA quantization]]): they span routes *and* disagree even within a route. DyQ-VLA (route 3) varies the *bit-width* dynamically; QuantVLA and Ω-QVLA (both route 2, both DuQuant-based, on the same two models) keep fixed precision but **make opposite choices on DiT attention** — Ω-QVLA quantizes the whole DiT uniformly (W4A4), QuantVLA selectively keeps DiT attention FP16 (W4A8) to preserve integer-GEMM deployability. So "VLA quantization" is an application domain that cuts across the routes, not a fourth method axis — keep the *method* taxonomy (format / distribution / dynamic) separate from the *domain* taxonomy (LLM / VLM / diffusion-DiT / VLA).

## Main subthemes likely to grow
- FP8 format design
- distribution reshaping and scaling
- rotation-based outlier suppression (orthogonal transforms; SVD vs Hadamard; block-wise)
- activation outliers in LLM quantization
- low-bit training stability
- calibration and scaling (incl. per-step calibration for diffusion / DiT; interface calibration à la QuantVLA ATM/OHB)
- precision versus dynamic range tradeoffs
- hardware-aware numerical representation design; integer-GEMM-preserving layouts vs accuracy-maximal fake-quant
- dynamic / input-conditioned bit allocation and the cost of the runtime control loop
- VLA / embodied-model quantization (domain signals as sensitivity proxies; action-head sensitivity)
- ultra-low-bit / binarization: multi-basis binary decomposition, native Boolean / latent-free training, ≤2-bit codebook structure (MBOK; BitNet / OneBit / BOLD lineage)
- KV-cache quantization: online data-oblivious vector quantization, unbiased inner-product estimators, distortion-rate bounds (TurboQuant; QJL / PolarQuant / KIVI lineage); interaction with token pruning ([[Visual token budget - pruning vs compression]])

## Open questions
- When do fixed FP8 formats become the bottleneck?
- How much can be gained by better numeric representation design rather than better scaling alone?
- Which low-bit strategies are mainly about inference efficiency, and which fundamentally affect training dynamics?
- How should one compare vendor-specific proposals against more standardized formats?
- When is quantization best framed as a representation-design problem versus a distribution-transformation problem?
- Which outlier-handling strategies generalize beyond SmoothQuant-style scaling? (rotation methods are the current answer)
- When does **dynamic (input-conditioned) bit allocation** pay for its runtime control overhead, versus a well-tuned static config?
- **Does VLA low-bit fundamentally need mixed precision, and must DiT attention stay FP?** DyQ-VLA argues dynamic bits; QuantVLA keeps DiT attention FP16 for deployability; Ω-QVLA quantizes it uniformly. Likely architecture- and deployment-dependent.
- Do LLM-PTQ insights (e.g. activation outliers) transfer to **VLA / embodied policies**, where errors compound through a closed observation→action loop?
- **How low can weights go, and what does it cost?** MBOK puts 2-bit (2 Boolean kernels) within ~1.2 ppl of FP16 on LLaMA-7B and 3-bit ≈ FP — but weight-only, on GPUs riding an INT1 kernel path. Is the ≤2-bit regime a training problem (latent-free optimization) or a representation problem (multi-basis codebooks)?
- **Weights vs KV cache.** TurboQuant makes 2.5–3.5-bit KV nearly free for text; do the same bit-widths hold for long multi-view embodied contexts under closed-loop error compounding, and do KV-bit savings compose multiplicatively with visual-token pruning/compression?

## Related
- [[VLA quantization]]
- [[Luo et al. - Ascend HiFloat8 Format for Deep Learning]]
- [[Xiao et al. - SmoothQuant Accurate and Efficient Post-Training Quantization for Large Language Models]]
- [[Lin et al. - DuQuant Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs]]
- [[Zheng et al. - DyQ-VLA Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models]]
- [[Xu et al. - QVLA Not All Channels Are Equal in Vision-Language-Action Models Quantization]]
- [[Zhang et al. - QuantVLA Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models]]
- [[Wang et al. - Omega-QVLA Robust Quantization for Vision-Language-Action Models via Composite Rotation and Per-step Scaling]]
- [[Tran and Nguyen - Highly Efficient and Effective LLMs with Multi-Boolean Architectures]] — Route 4 (Boolean-kernel decomposition + native Boolean training); first ≤2-bit source
- [[Zandieh et al. - TurboQuant Online Vector Quantization with Near-optimal Distortion Rate]] — first KV-cache (activations-at-rest) quantization source; distortion-rate theory behind rotation
