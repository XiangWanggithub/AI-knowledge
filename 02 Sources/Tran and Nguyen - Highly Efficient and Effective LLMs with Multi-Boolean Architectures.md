# Highly Efficient and Effective LLMs with Multi-Boolean Architectures (MBOK)

- Raw note: [[2026-09-04 - Tran and Nguyen - Highly Efficient and Effective LLMs with Multi-Boolean Architectures]]

## Metadata
- Type: source note
- Format: arXiv paper / **ICLR 2026 (Main Conference)**
- Authors: Ba-Hien Tran, Van Minh Nguyen
- Organization: **[[Huawei]] Paris Research Center** *(verified from PDF)* — the vault's second Huawei quantization source after [[Luo et al. - Ascend HiFloat8 Format for Deep Learning|HiFloat8]]
- Date accessed: 2026-09-04
- Original URL: https://arxiv.org/abs/2505.22811 · OpenReview: https://openreview.net/forum?id=r0CH5dF3Se
- arXiv ID: 2505.22811 (v1 2025-05-28 → v5 2026-04-21)
- Open source: **no** (no code link on arXiv/OpenReview; none found by search, 2026-09-04)
- Verification status: method, main tables and latency table **hand-verified against the full PDF (v5, 42 pp)** on 2026-09-04
- Related: [[Model quantization]], [[Zandieh et al. - TurboQuant Online Vector Quantization with Near-optimal Distortion Rate|TurboQuant]] (ingested together — the vault's other sub-4-bit source, on KV cache), [[Xiao et al. - SmoothQuant Accurate and Efficient Post-Training Quantization for Large Language Models|SmoothQuant]], [[Lin et al. - DuQuant Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs|DuQuant]], [[VLA quantization]]
- Tags: #quantization #binarization #ultra-low-bit #boolean-training #qat #knowledge-distillation #iclr2026 #huawei

## Summary
MBOK is the vault's **first binarization / ultra-low-bit (≤2-bit) source** and the first that does not fit the three existing routes of [[Model quantization]]. It represents every LLM linear layer as a **sum of K Boolean kernels** — K binary (±1) matrices, each scaled by its own rank-1 FP factor `s_out s_inᵀ` obtained by successive SVID on the residual — so K kernels cost ≈ K bits/weight. The kernels are then **trained directly in the Boolean domain**: instead of keeping FP16 latent weights and pushing STE-approximated gradients through `sign(·)`, a Boolean loss signal is accumulated per weight and the bit is **flipped** when the signal says so (extending Nguyen et al. 2024, *BOLD*). Knowledge is transferred from the FP model by data-free successive extraction followed by KD finetuning of **only the last kernel + scalings**; a greedy allocator picks K per layer under an average-bit budget (fractional bits allowed).

With 2 kernels (2-bit weights, FP16 activations) LLaMA-7B reaches ppl 6.83 vs FP16 5.68 — versus 15.34 for OmniQuant-2bit and 8.48 for OneBit — and 3 kernels essentially match FP16. On an A100 with BitBLAS INT1 kernels, LLaMA-13B linear layers run up to **8.7× faster than FP16**, while the 2-bit vector-quantization baselines (QuIP#, QTIP) run *slower* than FP16.

## Key claims
1. **Latent FP weights are the real cost of binarization, not the 1-bit weights.** QAT binarization (OneBit, MoS) needs a 16-bit latent copy plus two Adam momenta per parameter and STE proxy gradients that oscillate; training natively in Boolean space needs 1 bit + one momentum and no gradient approximation for the Boolean kernels.
2. **One binary kernel is too weak; a few residual kernels suffice.** SVID (rank-1 scaled sign) is Frobenius-optimal among rank-1 magnitude factors (Prop. 4.3) but cannot capture large pretrained matrices; 2–3 successive-residual kernels close most of the gap to FP16 (§6.1, Fig. 1).
3. **Only the last kernel needs finetuning.** Lower-order kernels carry the bulk of the signal and perturbing them cascades into higher-order residuals; finetuning only the last kernel gives the lowest flip rate and best perplexity (§6.1.2) — echoing "1-bit delta" compression (Liu et al. 2024b).
4. **Native 1-bit weights are the accelerator-friendly path to ≤2 bits**, in contrast to 2-bit vector quantization whose codebook lookups make it slower than FP16 on GPUs (Table 4, App. G.11).

## Results (verified)
- **Table 2, K=2 ("2×1" bits), FP16 activations:** OPT-1.3B 16.13/16.61 ppl (FP 14.62/14.72); LLaMA-7B **6.83/8.53** (FP 5.68/7.08; MoS-1b 7.97; OneBit-1b 8.48; OmniQuant-2b 15.34; OPTQ-2b collapses ~1.9e3); LLaMA-13B **6.17/7.88** (FP 5.09/6.61; MoS 7.16; OmniQuant-2b 13.43). Zero-shot avg LLaMA-7B 58.76 vs FP 64.06 vs OmniQuant-2b 48.17.
- **3 kernels ≈ FP** on OPT-125M…6.7B (Fig. 1, Table 3), Pareto-dominating RTN-3bit and OPTQ-3bit at equal size.
- **Latency, measured (Table 4; A100, batch 1, BitBLAS INT1×FP16):** LLaMA-13B linear layers 3.25× / 8.40× / **8.70×** vs FP16; QuIP# 0.27–0.69×, QTIP 0.08–0.09× (both slower than FP16).
- **Finetuning memory (Fig. 11, OPT-6.7B):** substantially below MoS — 1 bit + one 16-bit momentum per weight vs 16-bit latent + two momenta.
- Not measured: activation quantization; end-to-end model latency (only per-layer); any native Boolean hardware.

## Why it matters
1. **A fourth route for [[Model quantization]].** The topic's three routes intervene on the *format* (HiFloat8), the *distribution* (SmoothQuant → DuQuant → Ω-QVLA/QuantVLA), or the *runtime bit allocation* (DyQ-VLA / QVLA). MBOK intervenes on the **parameterization and the optimizer**: the weight *is* a set of Boolean kernels, and training happens in Boolean space. That is a different lever — "change what a weight is and how it is optimized," not "change how you round an FP weight" — so it opens a **Route 4: Boolean-kernel decomposition + native Boolean training** rather than extending Routes 1–3.
2. **Redraws the vault's bit floor.** Every previous source bottoms out at ~4-bit averages (W4A4/W4A8; QVLA's 0/2-bit channels are exceptions inside a ~4-bit average). MBOK's headline is a **2-bit average** with 3-bit ≈ FP, and it beats 2-bit PTQ (OPTQ/OmniQuant) and 1-bit QAT (OneBit/MoS) simultaneously — the first evidence in the vault of what the ≤2-bit regime costs and buys.
3. **A rare honest data point on the real-kernel axis.** The VLA-quant cluster's most-missed number is wall-clock; MBOK reports *measured* per-layer latency on a stock INT1 kernel path (BitBLAS) and shows that "smarter" 2-bit VQ is slower than FP16 — the same lesson as the cluster's "compression ratio ≠ speedup; layout and kernel availability decide" (cf. QVLA's unsubstantiated 1.49× vs DyQ-VLA's CUTLASS-measured one).
4. **Structural link, not just a number:** the multi-kernel sum is *not* uniform INT-K. Each kernel carries its own rank-1 scaling, so the reconstructed value grid is non-uniform and position-dependent — a structured, cheap codebook. This is the mechanistic reason it beats OmniQuant-2bit at equal bits, and it sits between scalar quantization (uniform grid) and vector quantization (arbitrary codebook) in expressivity while keeping scalar-quant kernels.

## What feels strong
- **Correct diagnosis of the cost center.** Attacking latent weights + STE rather than squeezing another bit is a genuinely different angle, and the finetuning-memory comparison (Fig. 11) shows it pays.
- **Data-free init + last-kernel-only finetuning** is a practical recipe: successive SVID needs no data, and the finetuning surface is tiny.
- **Kernel allocation with fractional bits** (PWCCA importance × residual × size, greedy) is a clean, reusable budgeter — the static-allocation idea of QVLA/HAWQ, applied to "how many binary kernels" instead of "which bit-width."
- **Real latency measured, with an unflattering VQ comparison included.**

## What feels limited
- **Weight-only, activations FP16** — so the 8.7× is a weight-bandwidth win in linear layers, not an end-to-end W-and-A story; not comparable to W4A4-class rotation methods.
- **No code**, and the Boolean optimizer's plasticity/Hebbian β_t schedule (App. A) is heuristic — reproducibility rests on the appendix.
- **Latency is per-layer at batch 1** (the memory-bound regime that most flatters weight compression); end-to-end decode throughput and prefill are not reported.
- **"No STE / no latent" is true only for the Boolean kernels**; scaling vectors and remaining FP parameters still use AdamW.
- **Awaits hardware.** The paper itself frames the full benefit as requiring native Boolean accelerators (§7); on GPUs it rides the INT1 path.
- **Naming.** "Multi-Boolean kernels" reads like logic circuits; it is a residual ±1 decomposition. The Boolean framing pays off in *training* (flip rule) and in the *hardware pitch* (XNOR/popcount), not in the representation.

## Ada's notes
The one-line demystification, kept because it was the sticking point when we read it: **a "Boolean kernel" is a `sign` matrix with a rank-1 FP scaling; "multi-Boolean" means K of them fitted successively to residuals; K kernels ≈ K bits.** Nothing about SAT or gates. What *is* Boolean-native is the optimizer: the weight update is a bit flip gated by an accumulated Boolean loss signal, which is why no FP16 latent copy is needed. The lower-order kernels are frozen after extraction and only the last kernel is trained — a nice inversion of the usual intuition that the "most important" part should be tuned.

Placement in the vault: MBOK does not extend Routes 1–3 of [[Model quantization]]; it is the seed of a **Route 4 (Boolean-kernel decomposition + native Boolean training)**, and the first ≤2-bit source. Together with [[Zandieh et al. - TurboQuant Online Vector Quantization with Near-optimal Distortion Rate|TurboQuant]] (ingested the same day) it also marks a second cut the topic page lacked: *what* is being quantized — weights (everything so far, incl. MBOK) vs the **KV cache** (TurboQuant).

On the "real kernel" axis the vault has been tracking across the VLA-quant cluster, MBOK is a positive example: measured latency on a public kernel library, plus the inconvenient finding that 2-bit VQ (QuIP#/QTIP) is *slower* than FP16 on GPU — worth remembering whenever a paper equates compression with speed.

## Questions worth following up
1. **End-to-end numbers.** Per-layer 8.7× at batch 1 → what does a full LLaMA-13B decode step gain, and how much survives at batch >1 or in prefill?
2. **Does native Boolean training transfer to VLAs?** The VLA cluster protects the action interface; would a flip-trained Boolean action head be stable in closed loop, or is the Boolean noise floor too high for continuous control?
3. **Compose with rotation.** SVID is applied to raw `W`; would a DuQuant/SVD·Hadamard-flattened `W` need fewer kernels for the same residual (allocation × reshaping are orthogonal, as noted on the [[VLA quantization]] page)?
4. **Activations.** The paper leaves them FP16; a Boolean-native activation path (BOLD-style) would be the real test of the "native Boolean hardware" pitch.
5. **RaBiT (arXiv:2602.05367)** claims residual-aware binarization *training*; is it MBOK with a different optimizer, or a return to latent weights?

## Possible downstream vault work
- Add **Route 4 (Boolean-kernel decomposition + native Boolean training)** to [[Model quantization]] and record the new **2-bit floor**. *(done in this ingest)*
- Extend [[Huawei]] with a second quantization line (Paris Research Center; Boolean training). *(done)*
- Candidate follow-ups: **BOLD** (Nguyen et al. 2024, the parent framework), **OneBit / MoS** (the QAT-with-latent baselines), **BitNet b1.58**, **RaBiT**; and, for the VQ contrast, **QuIP# / QTIP**.
