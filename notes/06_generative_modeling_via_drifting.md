# 06 · Deng et al. 2026 — Generative Modeling via Drifting

- arXiv：<https://arxiv.org/abs/2602.04770>（v2, 28 pages, CC-BY 4.0）
- 作者：Deng, M., Li, H., Li, T., Du, Y., He, K.（MIT + Harvard）
- 日期：7.19

## 核心视角

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

## 反对称性 → 均衡

**Prop. 3.1**：若 $V_{p,q}(x) = -V_{q,p}(x)\ \forall x$，则 $q = p \Rightarrow V_{p,q}(x) = 0$。

逆命题（$V \approx 0 \Rightarrow q \approx p$）不成立，仅在核形式 + 非退化假设下给出识别性 heuristic（Appx C.1）。

## 训练目标（Fixed-point MSE）

固定点条件：$f_{\hat\theta}(\epsilon) = f_{\hat\theta}(\epsilon) + V_{p,q_{\hat\theta}}(f_{\hat\theta}(\epsilon))$。

$$
\mathcal{L}(\theta) = \mathbb{E}_\epsilon
\Bigl\| f_\theta(\epsilon) - \operatorname{sg}\bigl[f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))\bigr] \Bigr\|^2
$$

- 数值上等于 $\mathbb{E}\|V(f_\theta(\epsilon))\|^2$，但 stop-grad 阻止梯度穿过 $V$（$V$ 内含 $q_\theta$）。
- 结构与 SimSiam（Chen & He 2021）、Consistency Training（Song & Dhariwal 2023）一致：stop-grad 自蒸馏 MSE。

## Kernelized drifting field

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
- Kernel：$k(x, y) = \exp(-\|x - y\|/\tau)$（$\ell_1$ 型 exp，非 RBF）。
- 数值实现：logits $= -\|x - y\|/\tau$，沿 $y$ 做 softmax 归一化；再在 batch 内 $\{x\}$ 上做一次 softmax。两次归一化不破坏反对称性。
- 归一化后的 kernel 与 InfoNCE（Oord 2018）形式同构。

## 特征空间 drifting

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

## 训练时 CFG

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

## 训练流程（Alg. 1 摘要）

1. 采一批噪声 $\{\epsilon_i\}$，生成 $\{x_i = f_\theta(\epsilon_i)\}$。
2. 采一批真实样本 $\{y^+_j\}$；令 $\{y^-_j\} = \{x_j\}$（同 batch，on-policy）。
3. 计算 $V(x_i) = V^+_p(x_i) - V^-_q(x_i)$（在特征空间 $\phi$）。
4. 目标 $t_i = \operatorname{sg}(\phi(x_i) + V(\phi(x_i)))$。
5. 反传 $\|\phi(x_i) - t_i\|^2$ 到 $\theta$。

## 实验结果（要点）

| 数据集 | 空间 | NFE | FID |
|---|---|---|---|
| ImageNet 256×256 | latent（SD-VAE） | 1 | 1.54 |
| ImageNet 256×256 | pixel | 1 | 1.61 |
| CIFAR-10（unconditional） | pixel | 1 | — 表 2 |
| 2D toy | pixel | 1 | 可视化 §5.1 |
| Robotic control | policy space | 1 | §5.3 |

主 claim：ImageNet 256×256 单步生成 FID 1.54 / 1.61 为 1-NFE SOTA。

## 与 Song 2021 的比较

| 维度 | Song 2021（SDE） | Drifting |
|---|---|---|
| 演化对象 | 样本沿物理时间 $t$ | pushforward 分布沿训练迭代 $i$ |
| Drift 定义 | $f(x,t) - g(t)^2 \nabla_x \log p_t(x)$ | $V_{p,q}(x)$，attraction − repulsion |
| 目标信息 | Score $\nabla_x \log p_t$（点估计） | 两分布的成对交互（kernel） |
| 训练目标 | Denoising score matching | Fixed-point MSE + stop-grad |
| 推理 | Reverse SDE / PF-ODE，多步 | 单次前向 $f_\theta(\epsilon)$，1-NFE |
| 均衡条件 | $s_\theta \to \nabla \log p_t$（点态） | $V = 0$（分布态） |
| Drift 的角色 | Inference-time dynamics | Training-time dynamics |

## 核心符号速查

| 符号 | 含义 |
|---|---|
| $f_\theta$ | 一步生成器 |
| $p_\epsilon$ | 输入噪声分布 |
| $q_\theta = f_{\theta\#}\,p_\epsilon$ | 生成分布 |
| $p$ | 真实数据分布 $p_{\text{data}}$ |
| $V_{p,q}$ | drifting field |
| $V^+_p, V^-_q$ | attraction / repulsion 分量 |
| $k(x, y)$ | kernel，$\exp(-\|x-y\|/\tau)$ |
| $\phi$ | 预训练自监督 encoder |
| $\alpha$ | CFG 强度参数 |
| $\operatorname{sg}[\cdot]$ | stop-gradient |

---

状态：主文完
