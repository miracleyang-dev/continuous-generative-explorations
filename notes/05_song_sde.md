# 05 · Song et al. 2021 — Score-Based Generative Modeling through SDEs

- 出处：ICLR 2021（outstanding paper）
- arXiv：<https://arxiv.org/abs/2011.13456>
- 日期：7.19

## 阅读结论

- **核心问题**：NCSN 与 DDPM 看似是两套离散模型，能否用一个连续框架统一训练、采样和 likelihood。
- **核心方法**：forward SDE 定义连续加噪路径；学习所有时刻的边缘 score 后，即可写出 reverse-time SDE。
- **关键桥梁**：同一组边缘分布还对应 probability flow ODE，因此随机采样、确定性采样和精确 likelihood 共用一个 score network。
- **历史位置**：VE-SDE 连到 NCSN，VP-SDE 连到 DDPM，PF-ODE 再连到 Flow Matching 的 velocity 视角。

## 1. Forward 与 reverse-time SDE

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
- 这条式子从 $T$ 积到 $0$，因此数值实现中的 $dt<0$；若改用递增的反向时间变量，drift 的符号也要一起变换。

## 2. Probability flow ODE

同一族 $p_t$ 存在等价的确定性 ODE：

$$
dx = \Bigl[f(x,t) - \tfrac12 g(t)^2\,\nabla_x \log p_t(x)\Bigr]\,dt
$$

推导：Fokker–Planck 允许两个不同 drift 满足同一 continuity equation 时保持 $p_t$ 一致。

用途：
- 用 ODE solver 采样（少步、无随机）。
- change-of-variables 计算精确 log-likelihood。
- 与 flow matching / rectified flow 中的 velocity field 直接对应。

## 3. 三类 SDE

### 3.1 VE-SDE：NCSN 的连续极限

$$
f(x,t)=0,
\qquad
g(t)=\sqrt{\frac{d\,\sigma^2(t)}{dt}}.
$$

方差随时间增加而均值不收缩，对应 SMLD / NCSN 的多尺度加性噪声。

### 3.2 VP-SDE：DDPM 的连续极限

$$
f(x,t)=-\frac{1}{2}\beta(t)x,
\qquad
g(t)=\sqrt{\beta(t)}.
$$

信号衰减同时注入噪声，总方差保持有界。对 VP-SDE 做 Euler-Maruyama 离散得到 DDPM forward chain，$\beta_t \approx \beta(t)\Delta t$。

### 3.3 sub-VP SDE：偏向 likelihood

$$
f(x,t)=-\frac{1}{2}\beta(t)x,
\qquad
g(t)=\sqrt{\beta(t)\left(1-e^{-2\int_0^t\beta(s)\,ds}\right)}.
$$

它与 VP-SDE 共享均值过程，但方差更小；论文用它取得更好的 likelihood。

## 4. 训练目标（连续时间 DSM）

$$
\mathcal{L}(\theta) = \mathbb{E}_t\Bigl[\lambda(t)\,
\mathbb{E}_{x_0}\mathbb{E}_{x_t\mid x_0}
\bigl\|s_\theta(x_t,t) - \nabla_{x_t}\log p_{0t}(x_t\mid x_0)\bigr\|^2\Bigr]
$$

- VE/VP 的转移核 $p_{0t}$ 是 Gaussian，$\nabla \log p_{0t}$ 有闭式。
- 原论文令 $\lambda(t)>0$，实践中通常取条件 score 目标平方范数的倒数，使不同时间的 loss 尺度平衡：

$$
\lambda(t) \propto
\left(\mathbb{E}\left[\left\|\nabla_{x_t}\log p_{0t}(x_t\mid x_0)\right\|_2^2\right]\right)^{-1}.
$$

不要把这篇论文的默认权重直接写成 $g(t)^2$；likelihood weighting 是相关后续分析中的另一种选择。

score 与 $\epsilon$-预测的换算：

$$
s_\theta(x_t, t) = -\epsilon_\theta(x_t,t)/\sigma_t
$$

### 4.1 训练流程

```text
repeat:
    t ~ Uniform[epsilon, T]
    x_0 ~ p_data
    x_t ~ p_0t(x_t | x_0)
    target = grad_x_t log p_0t(x_t | x_0)
    take gradient step on lambda(t) * ||s_theta(x_t, t) - target||^2
```

下界 $\epsilon>0$ 用来避开 $t=0$ 附近可能发散的条件 score。

## 5. 生成与评估

### 5.1 采样

- **Predictor**：反向 SDE 的数值 solver（Euler–Maruyama、reverse diffusion sampler、ancestral sampler 等）。
- **Corrector**：Langevin MCMC 步，用当前 $s_\theta$ 校正边缘。
- **PC sampler**：predictor 与 corrector 交替。SNR target 用 0.16（Algorithm 3）。
- **ODE sampler**：probability flow ODE + RK45 / DPM-Solver 类算法。

### 5.2 Log-likelihood

对 probability flow ODE 使用瞬时 change-of-variables：

$$
\log p_0(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot \tilde f_\theta(x_t, t)\,dt
$$

其中 $\tilde f_\theta(x,t) = f(x,t) - \tfrac12 g(t)^2 s_\theta(x,t)$。散度用 Hutchinson trace estimator 近似。

### 5.3 条件生成与逆问题

给定条件 $y$ 与似然 $p(y\mid x)$：

$$
\nabla_x \log p_t(x\mid y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y\mid x)
$$

将修改后的 score 代入反向 SDE / probability flow ODE。若观测模型的 likelihood score 可得，单个无条件 score model 可用于 inpainting、colorization 等逆问题；class-conditional generation 仍需条件 likelihood 或辅助分类器提供第二项。

## 6. 实验结果与局限

- CIFAR-10：FID 2.20，IS 9.89（NCSN++，continuous VE）。
- CIFAR-10 likelihood：2.99 bits/dim（DDPM++ cont. deep, sub-VP）。
- CelebA-HQ 1024×1024：首个 score-based 高分辨率结果。
- PC sampler 相比 predictor-only 稳定降低 FID。

局限：

- 反向过程仍是多步数值积分；solver、SNR、corrector 步数和容差引入较多超参数。
- ODE likelihood 需要估计 divergence，并同时积分状态与 log-density，计算显著慢于只做一次生成。
- 理论等价依赖真实 score；有限网络误差与离散误差会使 reverse SDE 和 PF-ODE 的结果产生差异。

## 7. 与前后工作的接口

- **NCSN**：VE-SDE 离散化，annealed Langevin 成为 corrector。
- **DDPM**：VP-SDE 离散化，$\epsilon$-prediction 是 score 的缩放参数化。
- **Flow Matching**：PF-ODE 的 drift 就是同一路径上的 velocity；FM 进一步允许不由 diffusion SDE 产生的路径。

## 8. 核心符号速查

| 符号 | 含义 |
|---|---|
| $f(x,t)$ | forward drift |
| $g(t)$ | forward diffusion coefficient |
| $p_t(x)$ | 时间 $t$ 的边缘 |
| $p_{0t}(x_t\mid x_0)$ | forward 转移核 |
| $s_\theta(x,t)$ | score network |
| $\epsilon_\theta(x,t)$ | 噪声预测网络 |
| $\bar W$ | 反向 Wiener 过程 |
| $\lambda(t)$ | DSM 的时间权重函数 |
