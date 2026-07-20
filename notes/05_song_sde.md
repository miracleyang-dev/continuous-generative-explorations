# 05 · Song et al. 2021 — Score-Based Generative Modeling through SDEs

- 出处：ICLR 2021（outstanding paper）
- arXiv：<https://arxiv.org/abs/2011.13456>
- 日期：7.19

## 框架

用一条 Itô SDE 描述加噪过程：

$$
dx = f(x,t)\,dt + g(t)\,dW,\qquad t\in[0,T]
$$

- $f(x,t)$：drift（漂移系数），控制期望方向。
- $g(t)$：diffusion（扩散系数，标量），控制注入噪声强度。
- $x_0 \sim p_{\text{data}}$，$x_T$ 近似先验（各向同性高斯）。
- 记边缘分布 $p_t(x)$，转移核 $p_{0t}(x_t\mid x_0)$。

反向时间过程（Anderson 1982）：

$$
dx = \bigl[f(x,t) - g(t)^2\,\nabla_x \log p_t(x)\bigr]\,dt + g(t)\,d\bar W
$$

- 反向 drift = 原 drift − $g(t)^2 \cdot \text{score}$。
- $\bar W$ 是反向时间的 Wiener 过程。

## Probability flow ODE

同一族 $p_t$ 存在等价的确定性 ODE：

$$
dx = \Bigl[f(x,t) - \tfrac12 g(t)^2\,\nabla_x \log p_t(x)\Bigr]\,dt
$$

推导：Fokker–Planck 允许两个不同 drift 满足同一 continuity equation 时保持 $p_t$ 一致。

用途：
- 用 ODE solver 采样（少步、无随机）。
- change-of-variables 计算精确 log-likelihood。
- 与 flow matching / rectified flow 中的 velocity field 直接对应。

## 两类 SDE 具体化

| 名称 | $f(x,t)$ | $g(t)$ | 离散对应 |
|---|---|---|---|
| VE-SDE | $0$ | $\sqrt{d[\sigma^2(t)]/dt}$ | NCSN（Song & Ermon 2019/2020） |
| VP-SDE | $-\tfrac12 \beta(t) x$ | $\sqrt{\beta(t)}$ | DDPM（Ho et al. 2020） |
| sub-VP | $-\tfrac12 \beta(t) x$ | $\sqrt{\beta(t)\bigl(1 - e^{-2\int_0^t\beta(s)ds}\bigr)}$ | — |

对 VP-SDE 做 Euler–Maruyama 离散得到 DDPM 的 forward chain，$\beta_t \approx \beta(t)\Delta t$。

## 训练目标（Denoising Score Matching）

$$
\mathcal{L}(\theta) = \mathbb{E}_t\Bigl[\lambda(t)\,
\mathbb{E}_{x_0}\mathbb{E}_{x_t\mid x_0}
\bigl\|s_\theta(x_t,t) - \nabla_{x_t}\log p_{0t}(x_t\mid x_0)\bigr\|^2\Bigr]
$$

- VE/VP 的转移核 $p_{0t}$ 是 Gaussian，$\nabla \log p_{0t}$ 有闭式。
- $\lambda(t) = g(t)^2$：loss 变为负 log-likelihood 的上界（Theorem 1）。

score 与 $\epsilon$-预测的换算：

$$
s_\theta(x_t, t) = -\epsilon_\theta(x_t,t)/\sigma_t
$$

## 采样

- **Predictor**：反向 SDE 的数值 solver（Euler–Maruyama、reverse diffusion sampler、ancestral sampler 等）。
- **Corrector**：Langevin MCMC 步，用当前 $s_\theta$ 校正边缘。
- **PC sampler**：predictor 与 corrector 交替。SNR target 用 0.16（Algorithm 3）。
- **ODE sampler**：probability flow ODE + RK45 / DPM-Solver 类算法。

## Log-likelihood

对 probability flow ODE 使用瞬时 change-of-variables：

$$
\log p_0(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot \tilde f_\theta(x_t, t)\,dt
$$

其中 $\tilde f_\theta(x,t) = f(x,t) - \tfrac12 g(t)^2 s_\theta(x,t)$。散度用 Hutchinson trace estimator 近似。

## 条件生成

给定条件 $y$ 与似然 $p(y\mid x)$：

$$
\nabla_x \log p_t(x\mid y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y\mid x)
$$

将修改后的 score 代入反向 SDE / probability flow ODE，不重训模型即可做 class-conditional、inpainting、colorization、inverse problems。

## 实验结果（要点）

- CIFAR-10：FID 2.20，IS 9.89（NCSN++，continuous VE）。
- CIFAR-10 likelihood：2.99 bits/dim（DDPM++ cont. deep, sub-VP）。
- CelebA-HQ 1024×1024：首个 score-based 高分辨率结果。
- PC sampler 相比 predictor-only 稳定降低 FID。

## 核心符号速查

| 符号 | 含义 |
|---|---|
| $f(x,t)$ | forward drift |
| $g(t)$ | forward diffusion coefficient |
| $p_t(x)$ | 时间 $t$ 的边缘 |
| $p_{0t}(x_t\mid x_0)$ | forward 转移核 |
| $s_\theta(x,t)$ | score network |
| $\epsilon_\theta(x,t)$ | 噪声预测网络 |
| $\bar W$ | 反向 Wiener 过程 |
| $\lambda(t)$ | loss 权重函数 |

---

状态：主文完
