# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

- Raw note: [[2026-09-04 - Zandieh et al. - TurboQuant Online Vector Quantization with Near-optimal Distortion Rate]]

## Metadata
- Type: source note
- Format: arXiv paper / **ICLR 2026** (venue per third-party sources; not on arXiv page — medium confidence)
- Authors: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
- Organization: **Google Research + New York University + Google DeepMind** *(verified from PDF)*
- Date accessed: 2026-09-04
- Original URL: https://arxiv.org/abs/2504.19874
- arXiv ID: 2504.19874 (v1 2025-04-28, only version)
- Open source: **no official code**; several third-party reimplementations exist (see raw note) — treat their latency claims as unverified
- Verification status: algorithms, theorem statements and all result tables **hand-verified against the full PDF (v1, 25 pp)** on 2026-09-04
- Related: [[Model quantization]], [[Tran and Nguyen - Highly Efficient and Effective LLMs with Multi-Boolean Architectures|MBOK]] (ingested together — the other sub-4-bit source, on weights), [[Lin et al. - DuQuant Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs|DuQuant]] (rotation lineage), [[Visual token budget - pruning vs compression]] (the other KV-cost axis), [[VLA quantization]]
- Tags: #quantization #kv-cache #vector-quantization #online #data-oblivious #rotation #information-theory #nearest-neighbor-search #iclr2026 #google

## Summary
TurboQuant is the vault's **first KV-cache / activations-at-rest quantization source** and its first *information-theoretic* quantization paper. It is an **online, data-oblivious vector quantizer** with a proof that it sits within a constant factor (≈2.7×, ≈1.45× at 1 bit) of the Shannon distortion-rate bound for **any** bit-width and dimension. The recipe is deliberately simple: **randomly rotate** the vector so every coordinate follows a known Beta (≈ Gaussian) distribution independent of the data, then apply a **precomputed optimal Lloyd-Max scalar quantizer per coordinate**. Because MSE-optimal quantizers are biased for inner products, a second variant spends its last bit on a **1-bit QJL sketch of the residual**, yielding an **unbiased** inner-product estimator — the property attention logits and nearest-neighbour search actually need.

On KV-cache compression it reaches **quality neutrality at 3.5 bits** (LongBench avg 50.06 = full 16-bit cache on Llama-3.1-8B-Instruct) and only −0.6 at **2.5 bits**, at ≥4.5× compression, while quantizing generated tokens too; on needle-in-a-haystack it matches full precision exactly at 4× compression where token-eviction methods (SnapKV, PyramidKV) degrade. On nearest-neighbour search it beats product quantization in recall while cutting indexing time from minutes/hours to ~1 ms.

## Key claims
1. **Random rotation makes the worst case tractable.** After rotation, coordinates of any unit vector are Beta/Gaussian-distributed and nearly independent in high `d` (Lemma 1), so *per-coordinate scalar* quantization with a fixed optimal codebook is near-optimal — no calibration data, no Hessians, no k-means at indexing time.
2. **MSE-optimal ≠ inner-product-optimal.** MSE quantizers are biased for `⟨y, x̂⟩` at low bit-widths (bias grows with the average inner product); composing a (b−1)-bit MSE quantizer with a 1-bit QJL residual sketch restores unbiasedness with near-optimal variance (Thm. 2).
3. **Near-optimality is provable.** Upper bounds `(√3π/2)·4^{−b}` (MSE) and `(√3π²‖y‖²/d)·4^{−b}` (inner product) vs lower bounds `4^{−b}` and `(‖y‖²/d)·4^{−b}` for *any* randomized quantizer (Thm. 3, Shannon + Yao minimax).
4. **Theory-backed quantizers beat heuristics on long context.** At equal 25% KV memory, PolarQuant and TurboQuant hold 0.995–0.997 needle recall vs KIVI 0.981, PyramidKV 0.895, SnapKV 0.858.

## Results (verified)
- **LongBench-E (Table 1), Llama-3.1-8B-Instruct avg:** Full 50.06 · KIVI-3b 48.50 · KIVI-5b 50.16 · PolarQuant-3.9b 49.78 · **TurboQuant-2.5b 49.44** · **TurboQuant-3.5b 50.06**. Ministral-7B: Full 49.89 · TurboQuant-2.5b 49.62.
- **Needle-in-a-haystack (Fig. 4), 4k–104k, KV ratio 0.25:** TurboQuant 0.997 = FP 0.997; PolarQuant 0.995; KIVI 0.981; PyramidKV 0.895; SnapKV 0.858.
- **NN search (§4.4):** best recall `1@k` at 2 and 4 bits on GloVe-200, OpenAI3-1536/3072 vs PQ and RabitQ; **indexing time 0.0007–0.0021 s vs PQ 37–494 s vs RabitQ 597–3957 s** (Table 2).
- **Bounds validated empirically** (Fig. 3): measured MSE / inner-product error lie between the proven upper and lower curves for `b = 1…5`.
- **Not reported:** attention wall-clock, decode throughput, memory-bandwidth savings measured on hardware; end-to-end latency on any serving stack.

## Why it matters
1. **A new *target* axis for [[Model quantization]].** Every earlier source quantizes **weights** (and, for W4A4-class methods, activations in flight). TurboQuant quantizes the **KV cache** — activations *at rest*, whose size scales with context length rather than parameter count. That is orthogonal to the topic's routes ("how you quantize") and deserves its own cut ("what you quantize"). It is also the axis that grows fastest in long-context and multi-view embodied settings.
2. **It supplies the theory behind Route 2's rotation trick.** QuaRot / [[Lin et al. - DuQuant Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs|DuQuant]] / Ω-QVLA rotate to *suppress outliers*, justified empirically. TurboQuant shows the same move — random rotation followed by per-coordinate scalar quantization — is **provably within 2.7× of the Shannon bound**, because rotation induces a known distribution for which a fixed Lloyd-Max codebook is optimal. Outlier suppression and distribution-inducing are two descriptions of one mechanism; TurboQuant is the distortion-rate account of it.
3. **Online / data-oblivious is a design constraint, not a convenience.** KV entries are produced during generation, so offline calibration (Hessians, k-means codebooks, DuQuant-style greedy rotations) is not available. TurboQuant's zero-calibration design is what lets it quantize *generated* tokens too, unlike KIVI/PolarQuant — the property that matters for streaming robot observations as much as for chat.
4. **Unbiasedness as a first-class objective.** Attention logits are inner products; a biased key quantizer shifts softmax temperature systematically (the same drift QuantVLA fights with ATM on the DiT side). Spending one bit on an unbiased residual sketch is a principled alternative to per-head temperature recalibration.

## What feels strong
- **Rare rigor:** matching upper and lower bounds, with experiments plotted against both.
- **Zero indexing/calibration cost** — orders of magnitude faster than PQ/RabitQ, and trivially vectorizable (rotation + per-coordinate lookup).
- **Honest baselines at equal memory** (25% KV) including eviction and scalar-quant methods, plus quantizing generated tokens.
- **Reusable beyond LLMs:** the same primitive serves vector databases and retrieval — one algorithm for two of the vault's efficiency concerns (KV cache, RAG-style memory).

## What feels limited
- **No wall-clock numbers.** Compression and indexing time are reported; attention latency, bandwidth savings and end-to-end throughput are not — the paper leaves the "real kernel" question open, and the speedups quoted online come from third-party reimplementations.
- **Small-model, text-only evidence:** two 7–8B instruct models on LongBench and needle; nothing on 70B-class, MoE, vision tokens or VLA contexts.
- **Dense rotation/QJL matrices** (`d×d` Gaussian) as stated; practical deployments will want structured rotations (Hadamard) and shared seeds — a gap between theorem and kernel.
- **Venue and code provenance are second-hand.** ICLR 2026 acceptance is asserted by third parties, not the arXiv page; there is no official implementation, so the many "TurboQuant" repos are interpretations.
- Which of the two quantizers is applied to keys vs values is not spelled out in the v1 text.

## Ada's notes
The clean way to file TurboQuant is as **the first entry on a second axis of [[Model quantization]]**: the routes say *how* (format / distribution / dynamic allocation / Boolean decomposition), this says *what* — the **KV cache**, i.e. activations at rest whose footprint scales with context length. That axis is invisible when every source quantizes weights, but it is exactly the cost that multi-view, history-carrying VLAs inflate: the [[Visual token budget - pruning vs compression]] page cuts the KV cost by reducing **token count**, TurboQuant cuts it by reducing **bits per token**; the two compose.

The second thing worth remembering is the bridge to Route 2. DuQuant learns a data-aware rotation to flatten outliers; TurboQuant uses a *random* rotation and proves that, once coordinates are Beta/Gaussian, a fixed optimal scalar codebook is near-optimal in the Shannon sense. So "rotate then scalar-quantize" is not a heuristic that happens to work — it has a distortion-rate guarantee — and the data-aware rotations of DuQuant/Ω-QVLA are refinements *inside* that guarantee, buying a smaller constant for a given weight matrix. Reading them together turns the vault's rotation lineage from an empirical trick into a principled design space.

On the honesty axis the vault has been keeping (measured speed vs claimed compression): TurboQuant is candid — it reports compression and indexing time and simply does not claim wall-clock speedups; the speedups attached to its name online are third-party. File it with [[Xu et al. - QVLA Not All Channels Are Equal in Vision-Language-Action Models Quantization|QVLA]] as "efficiency asserted, kernel unshown," but for a more defensible reason (the contribution is the quantizer, not a serving stack).

## Questions worth following up
1. **Keys vs values.** Does the paper (or a later version) use TurboQuant_prod for keys (inner products with queries) and TurboQuant_mse for values (averaged by softmax weights)? The split is natural but unstated in v1.
2. **Wall-clock on a real serving stack.** With a Hadamard rotation and fused dequant-into-attention kernels, what is the actual decode speedup at 32k–128k context vs FP16 KV — and does it survive on edge GPUs relevant to robots?
3. **Composition with token reduction.** Visual-token pruning/compression (EVS, π0.7 MEM) shrinks the KV cache along tokens; TurboQuant along bits. Are the savings multiplicative in practice, and does quantization noise interact with pruning decisions?
4. **VLA relevance.** Streaming multi-view observations with long histories are the embodied analogue of long context; does 2.5–3.5-bit KV hold under closed-loop error compounding (the [[VLA quantization]] concern), or does control need the 3.5-bit "neutral" point?
5. **Bridge to Route 2:** can the Beta-codebook argument bound how much a *learned* rotation (DuQuant) can improve over a random one for a given weight distribution?

## Possible downstream vault work
- Add a **"what is quantized" axis (weights vs KV cache)** to [[Model quantization]] with TurboQuant as the first KV-cache entry. *(done in this ingest)*
- Cross-link from [[VLA quantization]] and [[Visual token budget - pruning vs compression]] as the third efficiency axis (bits per KV token). *(done)*
- Candidate follow-ups: **QJL** (the authors' 1-bit sketch), **PolarQuant**, **KIVI / KVQuant** (the online KV-quant baselines), **QuaRot** (random-Hadamard rotation for weights — the natural bridge to Route 2), **Fast-TurboQuant** (arXiv:2606.21448).
