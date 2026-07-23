# 06 · Deng et al. 2026 — Generative Modeling via Drifting

- arXiv：<https://arxiv.org/abs/2602.04770>（v2, 28 pages, CC-BY 4.0）
- 作者：Deng, M., Li, H., Li, T., Du, Y., He, K.（MIT + Harvard）
- 日期：7.20

## 阅读结论

- **核心问题**：能否不蒸馏 SDE/ODE 轨迹，直接训练一个原生 1-NFE 生成器。
- **核心方法**：把训练迭代看作 pushforward 分布的演化，用真实样本吸引、生成样本排斥所构成的 drifting field 提供冻结目标。
- **关键机制**：反对称 field 保证 $p=q$ 时 drift 为零；fixed-point MSE 加 stop-gradient 把分布级信号转成网络更新。
- **主要代价**：一步推理把复杂度移到训练端，依赖大批量正负样本、成对 kernel 计算和强预训练特征 encoder。

## 1. 核心视角：训练时间的 pushforward 演化

生成器 $f_\theta : \mathbb{R}^C \to \mathbb{R}^D$，$x = f_\theta(\epsilon)$，$\epsilon \sim p_\epsilon$。
输出分布 $q = f_{\theta\#}\,p_\epsilon$。
训练迭代给出模型序列 $\{f_i\}$、分布序列 $\{q_i\}$。样本沿训练迭代演化：

$$
x_{i+1} = x_i + \Delta x_i,\qquad \Delta x_i = f_{i+1}(\epsilon) - f_i(\epsilon)
$$

抽象形式：

$$
x_{i+1} = x_i + V_{p, q_i}(x_i)
$$

$V_{p,q}: \mathbb{R}^d \to \mathbb{R}^d$ 为 drifting field；被漂移的对象是 pushforward 分布，
时间轴是训练迭代（区别于 SDE 中样本沿物理时间的演化）。

## 2. Drifting field 与均衡

### 2.1 反对称性

**Prop. 3.1**：若 $V_{p,q}(x) = -V_{q,p}(x)\ \forall x$，则 $q = p \Rightarrow V_{p,q}(x) = 0$。

逆命题（$V \approx 0 \Rightarrow q \approx p$）对任意 field 不成立。附录 C.1 在 full support、有限维基展开和 interaction vectors 线性独立等假设下给出充分条件；它是受限设定中的识别性论证，不是一般分布空间上的完整定理。

### 2.2 Fixed-point MSE

固定点条件：$f_{\hat\theta}(\epsilon) = f_{\hat\theta}(\epsilon) + V_{p,q_{\hat\theta}}(f_{\hat\theta}(\epsilon))$。

$$
\mathcal{L}(\theta) = \mathbb{E}_\epsilon
\Bigl\| f_\theta(\epsilon) - \operatorname{sg}\bigl[f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))\bigr] \Bigr\|^2
$$

- 数值上等于 $\mathbb{E}\|V(f_\theta(\epsilon))\|^2$，但 stop-grad 阻止梯度穿过 $V$（$V$ 内含 $q_\theta$）。
- 结构与 SimSiam（Chen & He 2021）、Consistency Training（Song & Dhariwal 2023）一致：stop-grad 自蒸馏 MSE。
- 因为 target 每步重新计算，这个更新更接近 fixed-point iteration；不能把实际梯度直接解释为对 $\mathbb{E}\|V\|^2$ 做完整梯度下降。

### 2.3 Kernelized drifting field

$$
V_{p,q}(x) = \mathbb{E}_{y^+ \sim p}\,\mathbb{E}_{y^- \sim q}\bigl[K(x, y^+, y^-)\bigr]
$$

Mean-shift 实例（Cheng 1995）：

$$
V^+_p(x) = \tfrac{1}{Z_p}\,\mathbb{E}_p\bigl[k(x, y^+)(y^+ - x)\bigr]
$$

$$
V^-_q(x) = \tfrac{1}{Z_q}\,\mathbb{E}_q\bigl[k(x, y^-)(y^- - x)\bigr]
$$

$$
V_{p,q}(x) := V^+_p(x) - V^-_q(x)
= \tfrac{1}{Z_p Z_q}\,\mathbb{E}_{p,q}\bigl[k(x, y^+)\,k(x, y^-)\,(y^+ - y^-)\bigr]
$$

- 语义：$x$ 被真实样本 $y^+$ 吸引，被生成样本 $y^-$ 排斥。
- Kernel：$k(x, y) = \exp(-\|x - y\|_2/\tau)$（未平方的 $\ell_2$ 距离指数核，不是常见的 squared-distance RBF）。
- 数值实现：logits $= -\|x - y\|/\tau$，沿 $y$ 做 softmax 归一化；再在 batch 内 $\{x\}$ 上做一次 softmax。两次归一化不破坏反对称性。
- 归一化后的 kernel 与 InfoNCE（Oord 2018）形式同构。
- 对 batch 内所有 query 与正负样本计算距离，朴素时间和显存复杂度为 $O(B^2)$；论文通过按类别分组和固定有效 batch size 管理该成本。

## 3. 特征空间与条件控制

### 3.1 特征空间 drifting

在预训练自监督 encoder $\phi(\cdot)$（如 MoCo v3）的表征空间做匹配：

$$
\mathcal{L}_\phi(\theta) = \mathbb{E}\,\bigl\| \phi(x) - \operatorname{sg}\bigl[\phi(x) + V(\phi(x))\bigr]\bigr\|^2
$$

多尺度多位置版本：对 ResNet 的多个 stage $\{\phi_j\}$ 分别计算再求和：

$$
\mathcal{L}_{\text{multi}} = \sum_j \mathbb{E}\,\bigl\| \phi_j(x) - \operatorname{sg}\bigl[\phi_j(x) + V(\phi_j(x))\bigr]\bigr\|^2
$$

- $\phi$ 仅训练时使用，推理仍单次前向。
- 与 perceptual loss 的区别：不需要配对目标；匹配的是 $\phi_\# q \to \phi_\# p$ 的分布层面。
- 与 latent-space generation 正交：$f_\theta$ 可为 pixel-space 或 SD-VAE latent-space 生成器；$\phi$ 是另一空间的 encoder。

### 3.2 训练时 CFG

用 mixture 负样本替换 $q$：

$$
\tilde q(\cdot\mid c) = (1 - \gamma)\,q_\theta(\cdot\mid c) + \gamma\,p_{\text{data}}(\cdot\mid \emptyset)
$$

令 $\tilde q(\cdot\mid c) = p_{\text{data}}(\cdot\mid c)$，解得：

$$
q_\theta(\cdot\mid c) = \alpha\,p_{\text{data}}(\cdot\mid c) - (\alpha - 1)\,p_{\text{data}}(\cdot\mid \emptyset),
\qquad \alpha = \tfrac{1}{1 - \gamma} \ge 1
$$

- 与标准 CFG（Ho & Salimans 2022）表达式一致：conditional 减 unconditional。
- CFG 作为训练时行为实现（Geng 2025b）：训练时随机采样 $\alpha$ 作为条件；推理时 $\alpha$ 任意指定，1-NFE 属性保留。

## 4. 训练流程（Algorithm 1 摘要）

1. 采一批噪声 $\{\epsilon_i\}$，生成 $\{x_i = f_\theta(\epsilon_i)\}$。
2. 采一批真实样本 $\{y^+_j\}$；令 $\{y^-_j\} = \{x_j\}$（同 batch，on-policy）。
3. 计算 $V(x_i) = V^+_p(x_i) - V^-_q(x_i)$（在特征空间 $\phi$）。
4. 目标 $t_i = \operatorname{sg}(\phi(x_i) + V(\phi(x_i)))$。
5. 反传 $\|\phi(x_i) - t_i\|^2$ 到 $\theta$。

## 5. 实验结果与局限

| 数据集 | 空间 | NFE | FID |
|---|---|---|---|
| ImageNet 256×256 | latent（SD-VAE） | 1 | 1.54 |
| ImageNet 256×256 | pixel | 1 | 1.61 |

主 claim：ImageNet 256×256 单步生成 FID 1.54 / 1.61。原笔记中“CIFAR-10 — 表 2”并不成立：论文表 2 是 ImageNet 的正负样本分配消融，因此这里不再列出没有对应结果的行。

### 5.1 关键消融

- 在固定有效 batch size 4096 下，增加每类正样本或负样本数量都持续改善 FID，说明 field 估计依赖批内覆盖。
- 特征 encoder 从 SimCLR / MoCo-v2 换成更强的 latent-MAE 后明显改善；增大 encoder 宽度和预训练轮数继续受益。
- 破坏 attraction/repulsion 的反对称结构会导致灾难性失败，实验与均衡条件一致。

### 5.2 局限

- 作者报告：ImageNet 上去掉特征 encoder 后无法训练成功；kernel 在原始空间可能接近“flat”，几乎不给有效 drift。
- 一步推理不等于低训练成本：方法需要大有效 batch、预训练 encoder、成对距离与长训练周期。
- $V=0 \Rightarrow p=q$ 的识别性仅在附录假设下论证；一般情形仍可能存在非目标平衡点。
- 截至本地 v2 预印本，结果集中在 ImageNet 与少量扩展任务，跨数据模态和规模的稳健性仍需验证。

## 6. 与 Song 2021 的比较

| 维度 | Song 2021（SDE） | Drifting |
|---|---|---|
| 演化对象 | 样本沿物理时间 $t$ | pushforward 分布沿训练迭代 $i$ |
| Drift 定义 | $f(x,t) - g(t)^2 \nabla_x \log p_t(x)$ | $V_{p,q}(x)$，attraction − repulsion |
| 目标信息 | Score $\nabla_x \log p_t$（点估计） | 两分布的成对交互（kernel） |
| 训练目标 | Denoising score matching | Fixed-point MSE + stop-grad |
| 推理 | Reverse SDE / PF-ODE，多步 | 单次前向 $f_\theta(\epsilon)$，1-NFE |
| 均衡条件 | $s_\theta \to \nabla \log p_t$（点态） | $V = 0$（分布态） |
| Drift 的角色 | Inference-time dynamics | Training-time dynamics |

## 7. 核心符号速查

| 符号 | 含义 |
|---|---|
| $f_\theta$ | 一步生成器 |
| $p_\epsilon$ | 输入噪声分布 |
| $q_\theta = f_{\theta\#}\,p_\epsilon$ | 生成分布 |
| $p$ | 真实数据分布 $p_{\text{data}}$ |
| $V_{p,q}$ | drifting field |
| $V^+_p, V^-_q$ | attraction / repulsion 分量 |
| $k(x, y)$ | kernel，$\exp(-\|x-y\|_2/\tau)$ |
| $\phi$ | 预训练自监督 encoder |
| $\alpha$ | CFG 强度参数 |
| $\operatorname{sg}[\cdot]$ | stop-gradient |
