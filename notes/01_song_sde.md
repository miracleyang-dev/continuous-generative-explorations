# 01 · Song 2021 — Score-Based Generative Modeling through SDEs

论文：<https://arxiv.org/abs/2011.13456>（ICLR 2021 outstanding）

日期：7.18

---

## 读这篇的动机

导师让我先啃 "Generative Modeling via Drifting"，但那篇一上来就默认你会 SDE
那一套语言（drift、reverse-time、probability flow ODE 都在飞），我硬啃了半天
公式对不上号。回头补 Song 2021 —— 它把 DDPM 和 score matching 都塞进同一个
SDE 框架，drift term $f(x,t)$ 就是从这里开始被大家挂在嘴边的。所以先补这篇，
再回去读 Drifting 应该顺很多。

## 到底在说什么

一句话：把"往图里逐步加噪、再学一个网络把它去掉"这件事，从 DDPM 那种
离散马尔可夫链改写成连续时间的 Itô SDE。forward 是加噪 SDE，reverse 是
Anderson (1982) 那条反向 SDE，只依赖 score $\nabla_x \log p_t(x)$。学一个
$s_\theta \approx \nabla_x \log p_t$ 就能反向采样。

顺手带出一个大礼：**probability flow ODE** —— 跟 SDE 有相同边缘分布的
确定性 ODE。这条 ODE 是后来 flow matching / rectified flow 的公共祖宗。
（我一开始以为 flow matching 是另起炉灶的东西，读到这里才发现不是。）

## 公式记忆点

**Forward SDE**

$$
dx = f(x,t)\,dt + g(t)\,dW
$$

$f$ 是漂移，$g$ 是扩散强度（标量）。$t\in[0,T]$，$x_0\sim p_{\text{data}}$，
$x_T$ 大致是高斯。

**Reverse-time SDE**（这个是 Anderson 82 的老结论，Song 借来用）

$$
dx = \bigl[f(x,t) - g(t)^2\,\nabla_x\log p_t(x)\bigr]\,dt + g(t)\,d\bar W
$$

只要 score 拿到了，反向 drift 就是 `原 drift - g^2 * score`，一步都不虚。
——整篇的枢轴其实就是这一行。

**Probability flow ODE**

$$
dx = \bigl[f(x,t) - \tfrac12 g(t)^2\,\nabla_x\log p_t(x)\bigr]\,dt
$$

系数从 $g^2$ 变 $\tfrac12 g^2$，$dW$ 消掉。推导路子：Fokker–Planck 里
两个不同的 drift 只要满足同一条 continuity equation，就共享 $p_t$。
用处：

- 换成 ODE solver 采样，步数少、不含随机；
- change of variables 直接算 log-likelihood；
- flow matching 的 velocity field 就是这个 ODE 的 drift $v = f - \tfrac12 g^2 s_\theta$。

**两个具体化的 SDE**

- VE：$f=0$，$g(t)=\sqrt{d\sigma^2/dt}$。对应 NCSN。
- VP：$f=-\tfrac12\beta(t)x$，$g=\sqrt{\beta(t)}$。对应 DDPM。

VP 的 Euler–Maruyama 离散就是 DDPM。所以 DDPM ⊂ SDE 视角，
这个包含关系是这篇的说服力主要来源。

**训练目标（DSM）**

$$
\mathcal{L} = \mathbb{E}_t\bigl[\lambda(t)\,
  \mathbb{E}_{x_0,x_t}\|s_\theta(x_t,t) - \nabla_{x_t}\log p_{0t}(x_t|x_0)\|^2\bigr]
$$

VP/VE 的转移核 $p_{0t}(x_t|x_0)$ 是 Gaussian，所以 RHS 的 score 有闭式，
loss 里没有 GAN 那种对抗东西。$\lambda(t)=g(t)^2$ 时能让 loss 变成负 ELBO 上界
（Appx D，还没细看）。

## 我这次没搞懂的地方

- $\lambda(t)=g(t)^2$ ⇒ 变成 ELBO 上界，这个推导 Appx D 具体怎么走 —— **待补**。
- Predictor-corrector 里 SNR target 0.16 那个数是纯调参出来的还是有推导？
- Probability flow ODE 算 likelihood 用 Hutchinson trace estimator，
  高维图像下方差会不会失控？作者报的 bpd 数字稳定性怎么保证？
- Sub-VP 相比 VP 的收益 —— 是数值稳定，还是 likelihood 真的更紧？没看清。

（先记着，等读到附录再一条条勾掉）

## 跟 DDPM / flow matching 的接口

DDPM 侧：
- forward chain = VP-SDE 的 Euler-Maruyama，$\beta_t \approx \beta(t)\Delta t$。
- $\epsilon_\theta$ 和 $s_\theta$ 的换算：$s_\theta = -\epsilon_\theta / \sigma_t$。
  （做实现时容易搞混符号，标一下。）

Flow matching / rectified flow 侧：
- probability flow ODE 的 drift 就是 flow matching 的 velocity。
- flow matching 直接 MSE 回归 velocity，不必显式经过 score；且路径可自由挑
  （不必被 VP/VE 那两条 Gaussian path 锁住）。
- 一条线捋下来：Song SDE → PF-ODE → flow matching，就是同一件事情越写越干净。

导师那篇 Drifting 的 drifting field $V_{p,q}$ 和这里的 reverse drift 是不是
同一个东西？—— 目前先猜"不是"：这里是 **inference-time 演化样本**，
Drifting 是 **training-time 演化 pushforward**。共享一个"漂移"的语言，
但被漂移的对象不一样。待明天读完 02 再回来对齐。

## 实操 checklist

（等真要复现再展开，先占个位）

- MNIST + VP-SDE，small Unet，forward → reverse SDE 走通；
- 再加一版 probability flow ODE 采样（`torchdiffeq`），比 FID / steps；
- 有余力就算个 bpd 对齐文献；
- 最后把 VP path 换成 flow-matching path，直接对上 02。

---

**状态**：读完主文，附录未细看
**开始**：7.18
**下一步**：明天读 02（Drifting）
