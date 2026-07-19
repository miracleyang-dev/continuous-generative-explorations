# 01 · Song et al. 2021 — Score-Based Generative Modeling through SDEs

> Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
> *Score-Based Generative Modeling through Stochastic Differential Equations*.
> ICLR 2021 outstanding paper.
> arXiv: https://arxiv.org/abs/2011.13456

## 0 · 为什么读这篇

- 把 DDPM (Ho 2020) 和 score matching (Song 2019) 统一到 **SDE** 框架下。
- 引入 **probability flow ODE** —— diffusion → flow matching 的桥梁。
- Drift term $f(x,t)$ 出现在这里，跟导师论文 "Generative Modeling via Drifting"
  的术语直接对齐。**必读前置**。

## 1 · 一句话总结

把"往数据里逐步加噪 + 学一个反过来去噪的网络"这件事，从离散的马尔可夫链
（DDPM）抬升到**连续时间随机过程**：forward 是一条 Itô SDE，reverse 是它的
Anderson (1982) 反向 SDE，只依赖 score $\nabla_x\log p_t(x)$；用 denoising
score matching 学一个 $s_\theta(x,t)\approx\nabla_x\log p_t(x)$ 就能采样。
这个 SDE 视角同时给出一个**确定性**采样器 —— probability flow ODE，把
score-based 模型改写成一个 ODE，从此可以做精确似然、可逆编辑，也让后来的
flow matching / rectified flow 有了共同语言。

## 2 · 关键公式

### 2.1 Forward / Reverse SDE

$$
dx = f(x,t)\,dt + g(t)\,dW,\qquad t\in[0,T]
$$

- $f(x,t)$：**drift** 系数（漂移项），控制期望的移动方向。
- $g(t)$：**diffusion** 系数（扩散强度），标量，把 Brownian motion $W$ 注入。
- 数据 $x_0\sim p_{\mathrm{data}}$，终态 $x_T$ 近似高斯（先验）。

反向时间过程（Anderson 1982）：

$$
dx = \bigl[f(x,t) - g(t)^2\,\nabla_x\log p_t(x)\bigr]\,dt + g(t)\,d\bar W
$$

- $\bar W$ 是反向时间下的 Wiener 过程。
- **反向 drift = 原 drift − $g^2\cdot$ score**。这一步是全篇枢轴：一旦拿到 score，反向 SDE 完全确定。

### 2.2 Probability Flow ODE

对同一个 $p_t$，存在一个确定性 ODE 与 SDE 有**相同边缘分布**：

$$
dx = \Bigl[f(x,t) - \tfrac12 g(t)^2\,\nabla_x\log p_t(x)\Bigr]\,dt
$$

- 系数少了因子 $\tfrac12$、去掉了 $dW$ 项。推导用 Fokker-Planck：两个不同
  的 drift 只要满足 continuity equation 保持 $p_t$ 一致，就可以互换。
- 意义：整个 score-based 模型可以看成一个 **neural ODE**，从此可以：
  1) 用 ODE solver 采样（更少步数、无随机性）；
  2) 用 change-of-variables 精确算 log-likelihood；
  3) 与 flow matching / rectified flow 直接对齐 —— 它们本质上就在学这个 ODE 的 velocity。

### 2.3 两个典型 SDE 实例

| 名字 | $f(x,t)$ | $g(t)$ | 对应离散模型 |
|---|---|---|---|
| VE-SDE | $0$ | $\sqrt{\frac{d[\sigma^2(t)]}{dt}}$ | NCSN (Song & Ermon 2019/2020) |
| VP-SDE | $-\tfrac12\beta(t)x$ | $\sqrt{\beta(t)}$ | DDPM (Ho 2020) |

VP-SDE 在 $\beta(t)$ 取合适 schedule 时和 DDPM 的 $\alpha_t$ 直接对应，
证明了 DDPM 是 VP-SDE 的一个 Euler-Maruyama 离散化。

### 2.4 训练目标：Denoising Score Matching

$$
\mathcal{L}(\theta) = \mathbb{E}_{t}\Bigl[\lambda(t)\,\mathbb{E}_{x_0}\mathbb{E}_{x_t\mid x_0}
\bigl\|s_\theta(x_t,t)-\nabla_{x_t}\log p_{0t}(x_t\mid x_0)\bigr\|^2\Bigr]
$$

- $p_{0t}(x_t\mid x_0)$ 是 forward SDE 的转移核，对 VE/VP 都是 Gaussian，
  score 有闭式。所以 $\nabla_{x_t}\log p_{0t}$ 可以直接算，不需要 GAN 那种对抗训练。
- $\lambda(t)$ 是加权函数。选 $\lambda(t)=g(t)^2$ 让 loss 变成负 ELBO 的上界，
  跟 DDPM 的 $L_{\text{simple}}$ 精神一致。

## 3 · 主要贡献

1. **统一框架**：把 SMLD (score matching with Langevin dynamics) 和 DDPM 都写成
   一个通用 SDE 的两个特例（VE-SDE / VP-SDE），并给出 sub-VP 变体（初始方差更小）。
2. **Reverse SDE 采样器**：给出可以用 predictor-corrector 混合采样的通用配方
   —— predictor 是数值 SDE solver，corrector 是 Langevin MCMC。
3. **Probability flow ODE**：证明存在一个共享 $p_t$ 的确定性 ODE；可用于
   1) 快速无随机采样 2) 精确似然 3) 潜空间可逆编辑（latent interpolation / manipulation）。
4. **可控生成**：只要有 $\nabla_x\log p(y\mid x)$，就能通过修改反向 drift 做
   class-conditional / inpainting / colorization / inverse problems，无需重新训练。
5. **实证 SOTA**：在 CIFAR-10 上 FID 2.20（NCSN++），Inception score 9.89，
   同时在 $1024\times1024$ CelebA-HQ 上给出第一次 score-based 高分辨率结果。

## 4 · 我不理解的点

（读的过程中随时补，示例条目占位；读到就替换成实际疑问）

- [ ] $\lambda(t)$ 选 $g(t)^2$ 时"变成 ELBO 上界"的推导细节 —— 附录 D 具体在推什么？
- [ ] Corrector 步的 signal-to-noise ratio 目标值 (Alg. 3) 为什么定 0.16？纯经验还是有理由？
- [ ] Probability flow ODE 用来算 likelihood 时，Hutchinson trace estimator 的方差如何在实践里控制？
- [ ] Sub-VP 相比 VP 的实际收益是"数值稳定"还是"likelihood 更紧"？

## 5 · 跟 diffusion / flow matching 的连接

**跟 DDPM (Ho 2020)**
- VP-SDE 的 Euler-Maruyama 离散化 = DDPM 的 forward chain（附录 B 有直接的
  一一对应，$\beta_t = \beta(t)\cdot \Delta t$）。
- DDPM 学的 $\epsilon_\theta$ 和 SDE 学的 $s_\theta$ 相差一个 $-1/\sigma_t$ 的
  尺度：$s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sigma_t$。

**跟 Flow Matching (Lipman 2022) / Rectified Flow (Liu 2022)**
- Probability flow ODE 的 drift $v(x,t) := f(x,t) - \tfrac12 g(t)^2 s_\theta(x,t)$
  就是 flow matching 里的 velocity field。
- Flow matching 直接以 $v$ 为回归目标学一个"路径"，不再显式经过 score；
  相当于把 SDE 视角"压扁"到 ODE 视角，训练目标更简洁（MSE against target
  velocity），且路径可自由选（不一定是 VP/VE 的 Gaussian path）。
- 因此：**Song SDE → probability flow ODE → flow matching** 是同一件事情
  越来越干净的三种写法。读导师的 "Generative Modeling via Drifting" 时，
  可以对照它的 drifting field $V_{p,q}$ 和这里的 reverse drift 有什么共同点、
  又本质区别在哪里（**已知区别**：drifting model 是 training-time 演化 pushforward
  分布，SDE 是 inference-time 演化样本；先记着，读完再总结）。

## 6 · 复现想法

（如果打算写代码，记在这里；先不做）

初步 checklist（等真要动手时再展开）：

- [ ] MNIST + VP-SDE，Unet-small，先跑通 forward SDE → reverse SDE 采样；
- [ ] 加 probability flow ODE 采样（`torchdiffeq`），对比两者 FID / 采样步数；
- [ ] （可选）在同一模型上算 log-likelihood，验证与文献 bpd 数值一致；
- [ ] （可选）把 VP-SDE 换成 flow-matching path，观察 velocity 视角下差异 —— 这一步直接铺垫到导师的 drifting paper。

---

**状态**：未开始
**开始日期**：
**完成日期**：
