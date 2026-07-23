# 01 · Song & Ermon 2019 — Generative Modeling by Estimating Gradients of the Data Distribution (NCSN)

- 出处：NeurIPS 2019
- arXiv：<https://arxiv.org/abs/1907.05600>
- 日期：7.14

## 阅读结论

- **核心问题**：真实数据接近低维流形，直接估计数据分布的 score 不稳定，Langevin dynamics 也难以跨越低密度区。
- **核心方法**：用多个高斯噪声尺度平滑数据，只训练一个以噪声尺度为条件的 score network。
- **核心闭环**：多尺度 DSM 负责学习，annealed Langevin dynamics 负责从大噪声到小噪声逐级采样。
- **历史位置**：NCSN 把“多噪声尺度 + score 回归 + 随机采样”连成完整生成模型，是后来 VE-SDE 的离散前身。

## 1. 为什么学习 score

数据分布 $p_{\text{data}}(x)$ 通常未知，直接建模似然要处理配分函数 $Z$。**Score** 定义为对数密度的梯度：

$$
s(x) := \nabla_x \log p_{\text{data}}(x)
$$

好处：$Z$ 对 $x$ 求导后消失，score 不依赖归一化常数。用网络 $s_\theta(x)$ 拟合 score，再用 Langevin dynamics 采样。

## 2. 怎样学习 score

理想目标（Explicit Score Matching, ESM）：

$$
J_{\text{ESM}}(\theta) = \tfrac12\,\mathbb{E}_{p_{\text{data}}}\bigl[\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\bigr]
$$

未知 $\nabla \log p_{\text{data}}$。Hyvärinen 2005 证明它等价于（在温和条件下）：

$$
J_{\text{ISM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\Bigl[\tfrac12\|s_\theta(x)\|^2 + \operatorname{tr}(\nabla_x s_\theta(x))\Bigr]
$$

- 无需知道 $p_{\text{data}}$。
- 高维下 trace 项需 $O(d)$ 次反传，成本过高 → 用 **denoising score matching**（Vincent 2011）替代。

### 2.1 Denoising Score Matching（DSM）

给数据加噪：$\tilde x = x + \sigma\,\varepsilon,\ \varepsilon\sim\mathcal{N}(0,I)$。转移核

$$
q_\sigma(\tilde x \mid x) = \mathcal{N}(\tilde x; x, \sigma^2 I)
$$

$$
\nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2}
$$

DSM 目标：

$$
J_{\text{DSM}}(\theta; \sigma) = \tfrac12\,\mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{q_\sigma(\tilde x\mid x)}
\Bigl[\bigl\| s_\theta(\tilde x, \sigma) + \tfrac{\tilde x - x}{\sigma^2}\bigr\|^2\Bigr]
$$

Vincent 2011 定理：$J_{\text{DSM}}(\theta;\sigma) = J_{\text{ESM}}(\theta; q_\sigma) + \text{const}$，即 DSM 学的是**加噪分布** $q_\sigma$ 的 score，而非 $p_{\text{data}}$ 的 score。

## 3. 为什么单尺度方案失败

### 3.1 Langevin dynamics

给定 $s_\theta(x) \approx \nabla_x \log p(x)$，Langevin 迭代

$$
x_{t+1} = x_t + \tfrac{\alpha}{2}\,s_\theta(x_t) + \sqrt{\alpha}\,z_t,\qquad z_t \sim \mathcal{N}(0, I)
$$

$\alpha \to 0$、$T\to\infty$ 且满足正则条件时，迭代分布收敛到 $p(x)$。有限步长会产生离散误差；严格说可用 Metropolis-Hastings 校正，原论文在实践中省略该校正。

### 3.2 两个失败模式

作者观察到：直接学 $s_\theta(x)$（未加噪）+ Langevin 采样效果差。原因：

**(A) 流形假设**：真实数据近似落在低维流形 $\mathcal{M} \subset \mathbb{R}^d$ 上。则 $p_{\text{data}}$ 在流形外密度为 0，$\nabla \log p$ 未定义 / 训练无信号。

**(B) 低密度区**：训练样本几乎不出现在低密度区，$s_\theta$ 在这些区域估计不准；而 Langevin 从随机点起步，需穿越低密度区。多模分布之间过渡时被卡住。

## 4. NCSN：多尺度噪声

关键改动：不用单一 $\sigma$，用一族几何序列

$$
\sigma_1 > \sigma_2 > \cdots > \sigma_L,\qquad \sigma_{i+1}/\sigma_i = \gamma < 1
$$

$\sigma_1$ 足够大以覆盖模式之间的低密度区；$\sigma_L$ 足够小以保留细节。学一个 **条件 score network** $s_\theta(x, \sigma)$。

**多尺度 DSM loss**（Eq. 5）：

$$
\mathcal{L}(\theta) = \frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i)\,J_{\text{DSM}}(\theta;\sigma_i)
$$

选 $\lambda(\sigma) = \sigma^2$，代入得

$$
\mathcal{L}(\theta) = \frac{1}{2L}\sum_{i=1}^L \mathbb{E}_{p_{\text{data}}}\mathbb{E}_{q_{\sigma_i}}
\Bigl[\bigl\|\sigma_i\,s_\theta(\tilde x, \sigma_i) + \tfrac{\tilde x - x}{\sigma_i}\bigr\|^2\Bigr]
$$

- 直觉：$\sigma_i s_\theta$ 的量纲和 $\varepsilon$ 一致，各尺度贡献可比。
- 与 DDPM 的 $L_{\text{simple}}$ 精神一致（都在 MSE 一个"标准化后的目标"）。

### 4.1 训练流程

```text
repeat:
    x ~ p_data
    i ~ Uniform{1, ..., L}
    epsilon ~ N(0, I)
    x_tilde = x + sigma_i * epsilon
    take gradient step on lambda(sigma_i)
        * ||s_theta(x_tilde, sigma_i) + epsilon / sigma_i||^2
```

一次迭代只采一个噪声尺度，因此训练开销不随 $L$ 线性增长；网络通过条件归一化接收 $\sigma_i$。

### 4.2 Annealed Langevin Dynamics（采样）

从大 $\sigma_1$ 起步，逐步降到 $\sigma_L$：

**Algorithm 1**（简化版）：
```text
初始化 x0 ~ 均匀
for i = 1, ..., L:
    alpha_i = epsilon * sigma_i^2 / sigma_L^2
    for t = 1, ..., T:
        z ~ N(0, I)
        x <- x + (alpha_i / 2) * s_theta(x, sigma_i) + sqrt(alpha_i) * z
return x
```

- 大 $\sigma$ 阶段：分布光滑、模式之间可穿越。
- 小 $\sigma$ 阶段：refine 细节。
- 步长 $\alpha_i \propto \sigma_i^2$ 保证信噪比在各尺度一致。

### 4.3 网络与训练细节

- 网络：RefineNet（U-Net 系）作为 $s_\theta(x, \sigma)$，$\sigma$ 通过 conditional instance normalization 注入。
- 数据集：MNIST、CelebA、CIFAR-10。
- $L = 10$，$\sigma_1 = 1$，$\sigma_L = 0.01$（CIFAR-10）。

## 5. 实验结果与局限

- CIFAR-10：Inception Score 8.87，FID 25.32（NCSN）。
- 定性：多模生成不塌，可 inpainting。
- 后续 NCSNv2（Song & Ermon 2020）改进 $\sigma_i$ schedule 与 EMA，进一步降 FID。

局限需要与结果一起看：

- 原论文采样使用 $L=10$ 个噪声层、每层 $T=100$ 次 Langevin 更新，约需 1000 次网络评估，生成速度慢。
- 有限步长且不做 Metropolis-Hastings 校正，样本来自目标分布的近似而非严格无偏采样。
- 最小噪声 $\sigma_L$、步长 $\epsilon$ 和每层迭代数都需要调参；$q_{\sigma_L}$ 也只是逼近未加噪数据分布。

## 6. 与后续工作的接口

- **DDPM (Ho 2020)**：与 NCSN 用不同参数化（$\epsilon$-prediction）与不同加噪 schedule（VP 型），但 loss 本质是同一个 DSM 家族。
- **Song 2021（SDE）**：NCSN 对应 VE-SDE 的离散化：$f(x,t) = 0$，$g(t) = \sqrt{d\sigma^2/dt}$。$s_\theta(x, \sigma)$ 就是连续时间 $s_\theta(x, t)$。
- **PC sampler**：predictor 是数值 SDE solver，corrector 就是这里的 Langevin。

## 7. 核心符号速查

| 符号 | 含义 |
|---|---|
| $p_{\text{data}}$ | 数据分布 |
| $s_\theta(x, \sigma)$ | 条件 score network |
| $q_\sigma(\tilde x\mid x)$ | 高斯加噪核 $\mathcal{N}(x,\sigma^2 I)$ |
| $\sigma_1 > \cdots > \sigma_L$ | 几何噪声序列 |
| $\lambda(\sigma)$ | loss 权重，默认 $\sigma^2$ |
| $\alpha_i$ | 第 $i$ 个噪声尺度的 Langevin 步长 |
