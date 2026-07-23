# 04 · Ho & Salimans 2022 — Classifier-Free Diffusion Guidance (CFG)

- 出处：NeurIPS 2021 Workshop on DGMs and Applications
- arXiv：<https://arxiv.org/abs/2207.12598>
- 日期：7.17

## 阅读结论

- **核心问题**：classifier guidance 能提高条件一致性，但需要额外训练一个能识别加噪样本的分类器。
- **核心方法**：同一去噪网络同时学习 conditional 与 unconditional 预测，推理时做线性外推。
- **核心权衡**：guidance scale 越大，条件保真度通常越高、覆盖度越低；最佳 FID 与最佳 IS 不在同一点。
- **实际代价**：标准 CFG 每个采样步要跑 conditional 与 unconditional 两次前向，不能把“无分类器”理解为“无额外推理成本”。

## 1. 从 Classifier Guidance 出发

在条件生成 $p(x\mid y)$ 中，用 Bayes：

$$
\nabla_x \log p_t(x\mid y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y\mid x)
$$

第二项由一个**外部分类器** $p_\phi(y\mid x_t)$ 提供。带 guidance 权 $w$ 的采样使用：

$$
\hat s(x_t, y) = \nabla_x \log p_t(x_t) + w\,\nabla_x \log p_t(y\mid x_t)
$$

- $w > 1$：加强条件，提升 IS，降低 diversity。
- 需要单独训一个在**加噪数据**上的分类器 $p_\phi(y\mid x_t)$，代价高。

## 2. Classifier-Free Guidance

单一网络 $\varepsilon_\theta(x_t, t, y)$ 同时学**条件**和**无条件**两个模型：训练时以概率 $p_{\text{uncond}}$（典型 0.1–0.2）把 $y$ 替换为空标记 $\emptyset$。

推理时用两个前向组合出 guided 预测：

$$
\tilde\varepsilon_\theta(x_t, t, y) = (1 + w)\,\varepsilon_\theta(x_t, t, y) - w\,\varepsilon_\theta(x_t, t, \emptyset)
$$

或等价地（另一常见写法，$s := 1 + w$）：

$$
\tilde\varepsilon = \varepsilon_\theta(x_t, t, \emptyset) + s\,\bigl(\varepsilon_\theta(x_t, t, y) - \varepsilon_\theta(x_t, t, \emptyset)\bigr)
$$

### 2.1 与 score 的等价推导

$\varepsilon$ 与 score 的关系（DDPM 侧）：

$$
\varepsilon_\theta(x_t, t, y) = -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t} \log p_\theta(x_t\mid y)
$$

代入 guided 预测：

$$
\begin{aligned}
\tilde\varepsilon = -\sqrt{1-\bar\alpha_t}\,\Bigl[
&\nabla_{x_t}\log p_t(x_t\mid \emptyset) \\
&+ s\,\bigl(
\nabla_{x_t}\log p_t(x_t\mid y)
{}- \nabla_{x_t}\log p_t(x_t\mid \emptyset)
\bigr)
\Bigr].
\end{aligned}
$$

化简（假设 $p_t(x_t\mid\emptyset) = p_t(x_t)$）：

$$
\nabla_{x_t}\log \tilde p_t(x_t\mid y) = \nabla_{x_t}\log p_t(x_t) + s\,\bigl(\nabla_{x_t}\log p_t(x_t\mid y) - \nabla_{x_t}\log p_t(x_t)\bigr)
$$

进一步用 Bayes $\log p_t(x_t\mid y) = \log p_t(x_t) + \log p_t(y\mid x_t) - \log p_t(y)$，$\log p_t(y)$ 与 $x_t$ 无关，梯度消去：

$$
\nabla_{x_t}\log \tilde p_t(x_t\mid y) = \nabla_{x_t}\log p_t(x_t) + s\,\nabla_{x_t}\log p_t(y\mid x_t)
$$

**结论**：CFG 隐式实现了 $\nabla \log p_t(y\mid x_t)$ 的引导，$s$ 起 Classifier Guidance 里 $w$ 的作用（差一个基准偏移）。**不需要单独的分类器**。

### 2.2 隐含的分布形式

采样从

$$
\tilde p_t(x_t\mid y) \propto p_t(x_t)\,p_t(y\mid x_t)^s
$$

- $s = 1$：标准条件采样。
- $s > 1$：$p_t(y\mid x_t)$ 被"锐化"，样本更服从条件、模式塌缩加剧。
- $s = 0$：退化为无条件。

## 3. 训练与采样

### 3.1 训练算法

```text
输入: 数据集 {(x_0, y)}, 无条件概率 p_uncond
repeat:
    x_0, y ~ 数据
    y <- EMPTY  with prob p_uncond
    t ~ Uniform{1, ..., T}
    eps ~ N(0, I)
    x_t = sqrt(bar_alpha_t) x_0 + sqrt(1 - bar_alpha_t) eps
    take gradient step on || eps - eps_theta(x_t, t, y) ||^2
```

### 3.2 采样算法

```text
输入: 条件 y, guidance scale s
x_T ~ N(0, I)
for t = T, ..., 1:
    eps_cond = eps_theta(x_t, t, y)
    eps_uncond = eps_theta(x_t, t, EMPTY)
    eps_tilde = eps_uncond + s * (eps_cond - eps_uncond)
    根据 DDPM/DDIM 反向公式用 eps_tilde 得到 x_{t-1}
return x_0
```

- 推理时**每步两次前向**（cond + uncond），成本 ~2×。
- $s$ 可任意选，无需重训。

### 3.3 条件注入方式

- Class-conditional：$y$ = 类别 embedding，加入 timestep embedding 或 AdaGN。
- Text-to-image：$y$ = 文本 embedding，用 cross-attention 注入 U-Net；GLIDE、Stable Diffusion、Imagen 等后续系统沿用了这一范式。
- 无条件 token $\emptyset$：可学习的零向量或专用 embedding。

## 4. 实验结果与局限

- ImageNet 64×64：小幅 guidance 得到最佳 FID，更强 guidance 持续提高 IS，但 FID 随后变差，清楚展示 fidelity-diversity frontier。
- ImageNet 128×128：$T=256$、论文记号 $w=0.3$ 时 FID **2.43**；同表 ADM-G 为 2.97。
- $p_{\text{uncond}}=0.1$ 与 0.2 表现接近，0.5 的整条 IS/FID frontier 较差，说明无条件训练占比也需调参。

局限：

- 每步两次完整去噪网络前向；按相同网络比较速度时，优势会小于只看采样步数所得的印象。
- 大 guidance 通过牺牲覆盖度换取保真度，可能进一步压低少数模式和欠代表群体的出现概率。
- conditional 与 unconditional 估计来自同一网络并不保证外推后的 score 在有限模型误差下对应一个一致、归一化的概率密度。

## 5. 与其他技术的关系

- **Classifier Guidance**（Dhariwal & Nichol 2021）：CFG 的显式版本，需要外部 $p_\phi(y\mid x_t)$。
- **Score-based SDE**（Song 2021）：条件反向 SDE $\nabla \log p_t(x\mid y)$，CFG 是它的一个 practical 实例。
- **Drifting Model**（Deng et al. 2026）：把 CFG 写成训练时的负样本 mixture，目标分布满足

$$
q_\theta(\cdot\mid c) = \alpha\,p_{\text{data}}(\cdot\mid c) - (\alpha - 1)\,p_{\text{data}}(\cdot\mid\emptyset),
\qquad \alpha = \frac{1}{1-\gamma}.
$$

它与 CFG 都含“conditional 减 unconditional”的外推结构，但一个发生在 diffusion 推理阶段，另一个进入一步生成器的训练目标，不能视为同一算法。

## 6. 核心符号速查

| 符号 | 含义 |
|---|---|
| $y$ | 条件（类别 / 文本 embedding） |
| $\emptyset$ | 无条件 / 空 token |
| $p_{\text{uncond}}$ | 训练时把 $y$ 换成 $\emptyset$ 的概率 |
| $w,\ s = 1 + w$ | guidance scale |
| $\varepsilon_\theta(x_t, t, y)$ | 条件噪声预测 |
| $\varepsilon_\theta(x_t, t, \emptyset)$ | 无条件噪声预测 |
| $\tilde\varepsilon$ | guided 组合预测 |
| $\tilde p_t(x_t\mid y)$ | 隐含目标分布 $\propto p_t(x_t) p_t(y\mid x_t)^s$ |
