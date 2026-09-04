# Huawei（华为）

- **Type**: Entity（公司）
- **相关业务（本库范围）**: **Ascend（昇腾）** AI 加速器 / NPU 产品线，及深度学习数值格式与训练/推理软硬件栈。
- **本库依据**: [[Luo et al. - Ascend HiFloat8 Format for Deep Learning|Ascend HiFloat8]]（Yuanyong Luo 等）的署名机构；以及 [[Tran and Nguyen - Highly Efficient and Effective LLMs with Multi-Boolean Architectures|MBOK]]（**Huawei Paris Research Center**，Ba-Hien Tran & Van Minh Nguyen，ICLR 2026）的署名机构。

## 与本库的关联
- **HiFloat8 (HiF8)**：一种 8-bit 浮点格式，用可变字段 `dot field` 实现**渐变精度（tapered precision）**——试图用单一格式兼顾 FP8 E4M3/E5M2 各自的精度-动态范围权衡。
- 在 [[Model quantization]] 分类中归**表征设计（数值格式）路线**（Route 1 / FP8 一支）——本库量化簇里少见的"设计新数值格式"方向，与 SmoothQuant / DuQuant 的**分布重塑**路线互补。也是 [[VLA quantization]] 页指出的"尚无 VLA Route-1 例子"的上游母方法。
- **MBOK（Multi-Boolean Kernels，ICLR 2026）**：巴黎研究中心的**极低比特**路线——把每个线性层表示为 K 个 ±1 布尔核（各带秩-1 缩放）之和，并**在布尔域直接训练**（翻位规则，无 FP latent 权重、无 STE）；2-bit LLaMA-7B ppl 6.83 vs FP16 5.68，BitBLAS INT1 实测单层 8.7×。在 [[Model quantization]] 中开出**路线 4（布尔核分解 + 布尔域原生训练）**，是本库首个 ≤2-bit 源。与 HiF8 合起来，华为在本库量化簇里同时占据“格式设计”和“极低比特”两端。

## Related
- [[Luo et al. - Ascend HiFloat8 Format for Deep Learning]] — 本库所收其工作
- [[Model quantization]] — 表征设计 / FP8 路线
- [[Tran and Nguyen - Highly Efficient and Effective LLMs with Multi-Boolean Architectures]] — 本库所收其第二篇量化工作（路线 4，极低比特 / 布尔域训练）

## tags
#entity #org #hardware #quantization #fp8 #binarization #china
