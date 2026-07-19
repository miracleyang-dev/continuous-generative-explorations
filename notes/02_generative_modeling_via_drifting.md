# 02 · Generative Modeling via Drifting

论文：<https://arxiv.org/abs/2602.04770>（v2, 2026, CC-BY 4.0）
作者：Deng, M., Li, H., Li, T., Du, Y., He, K. — MIT + Harvard。

日期：7.19

---

## 读这篇的动机

昨天读完 Song 2021（见 01），今天正式开这篇。导师的主线论文，绕不开。
主 claim 很硬：ImageNet 256×256 **单步** FID 1.54 (latent) / 1.61 (pixel)。
如果站得住，这是近期 1-NFE 生成器里数字最好的一批之一。逐节拆一下。

## 一句话

不再让样本在**推理时**沿 SDE/ODE 演化到数据，而是让 pushforward 分布
$q = f_\# p_\epsilon$ 在**训练时**沿一个 drifting field $V_{p,q}$ 演化到
$p_{\text{data}}$。$V$ 取"真实样本吸引 − 生成样本排斥"的 mean-shift 形式，
loss 用 stop-grad 的 fixed-point MSE。训完 $f_\theta$ 就是一步生成器。

—— 视角切换的一句：**把 drift 从 inference dynamics 挪到 training dynamics**。
（昨天读 Song 时挂的那个悬念今天在这里落地了。）

## 公式记忆点

### 2.1 Pushforward，训练时演化

生成器 $f:\mathbb{R}^C\to\mathbb{R}^D$，$x = f(\epsilon)$，$\epsilon\sim p_\epsilon$。
输出分布记 $q = f_\# p_\epsilon$。SGD 走一步，模型序列 $\{f_i\}$、分布序列
$\{q_i\}$，样本被"漂移"：

$$
x_{i+1} = x_i + \Delta x_i, \quad \Delta x_i = f_{i+1}(\epsilon) - f_i(\epsilon)
$$

对应到抽象的 update：$x_{i+1} = x_i + V_{p,q_i}(x_i)$。

### 2.2 Drifting field + 反对称

Drifting field $V_{p,q}:\mathbb{R}^d\to\mathbb{R}^d$。

**Prop. 3.1**：如果 $V_{p,q}(x) = -V_{q,p}(x)\ \forall x$，则 $q=p \Rightarrow V=0$。

反向命题不成立 —— $V\approx 0 \Rightarrow q\approx p$ 只有在核形式 + 温和
non-degeneracy 假设下作者给了个 heuristic（Appx C.1）。这就是全文最"脆"的一块，
后面第 5 节要单独打问号。

### 2.3 Fixed-point loss

fixed point：$f_{\hat\theta}(\epsilon) = f_{\hat\theta}(\epsilon) + V_{p,q_{\hat\theta}}(f_{\hat\theta}(\epsilon))$。
写成 loss：

$$
\mathcal{L} = \mathbb{E}_\epsilon\bigl\|
f_\theta(\epsilon) - \operatorname{sg}\bigl[\,f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))\bigr]
\bigr\|^2
$$

数值上 = $\mathbb{E}\|V(f(\epsilon))\|^2$，但 stop-grad 保证梯度**不穿过 $V$**
（$V$ 里嵌了 $q_\theta$，穿过一个分布不好搞）。

血缘：跟 SimSiam (Chen & He 2021) 和 CT (Song & Dhariwal 2023) 是同一系
stop-grad + 自蒸馏 MSE。所以这不是全新的训练技术，是把它嫁接到了新对象上。

### 2.4 Kernel 化的 drifting field

$$
V_{p,q}(x) = \mathbb{E}_{y^+\sim p}\mathbb{E}_{y^-\sim q}[K(x,y^+,y^-)]
$$

作者取 mean-shift 型（Cheng 1995）：

$$
V^+_p(x) = \tfrac{1}{Z_p}\mathbb{E}_p[k(x,y^+)(y^+-x)],\quad
V^-_q(x) = \tfrac{1}{Z_q}\mathbb{E}_q[k(x,y^-)(y^--x)]
$$

$$
V_{p,q}(x) := V^+_p(x) - V^-_q(x)
= \tfrac{1}{Z_pZ_q}\mathbb{E}_{p,q}[k(x,y^+)k(x,y^-)(y^+-y^-)]
$$

直觉简单：$x$ 被 $y^+$ 拉过去，被 $y^-$ 推开。

- Kernel：$k(x,y) = \exp(-\|x-y\|/\tau)$，**$\ell_1$ 型 exp**，不是常见的 $\ell_2$ RBF。
  实现走 softmax（logits $= -\|x-y\|/\tau$ 沿 $y$ 归一化），再额外在 batch 的 $\{x\}$ 上做
  一次 softmax。
- 后面这个"batch-$\{x\}$ softmax" 我目前没看懂在归一化什么维度，见第 5 节 TODO。
- softmax 化之后跟 InfoNCE (Oord 2018) 同构 —— 视作 "对比学习 × mean-shift"。

### 2.5 特征空间 drifting

像素空间直接算 kernel 高维下没戏。搬到 encoder $\phi$（预训练自监督，比如 MoCo）
的特征空间：

$$
\mathbb{E}\|\phi(x) - \operatorname{sg}[\phi(x) + V(\phi(x))]\|^2
$$

多尺度多位置：对 ResNet 各 stage $\phi_j$ 都算一遍再加起来。
$\phi$ 只在训练时用，推理仍旧一次前向。

- 跟 perceptual loss 的区别：perceptual 要**配对目标** $\phi(x) - \phi(x_{\text{gt}})$；
  这里的回归目标是 $\phi(x) + V(\phi(x))$，不配对，匹配的是分布 $\phi_\# q \to \phi_\# p$。
- 跟 latent generation 是**正交**的：$f_\theta$ 可以是 pixel-space 或 SD-VAE latent-space；
  $\phi$ 是**另一个**空间的 encoder。两者独立选。（第一次读容易混，特别标一下。）

### 2.6 训练时 CFG

用 mixture 负样本分布：

$$
\tilde q(\cdot|c) = (1-\gamma)\,q_\theta(\cdot|c) + \gamma\,p_{\text{data}}(\cdot|\emptyset)
$$

要求 $\tilde q(\cdot|c) = p_{\text{data}}(\cdot|c)$，推出

$$
q_\theta(\cdot|c) = \alpha\,p_{\text{data}}(\cdot|c) - (\alpha-1)\,p_{\text{data}}(\cdot|\emptyset),
\quad \alpha = \tfrac{1}{1-\gamma}\ge 1
$$

精神跟标准 CFG (Ho & Salimans 2022) 一致，"conditional 减 unconditional"。
关键差别：**这里 CFG 是训练时行为**。作者的做法（CFG-conditioning, Geng 2025b）：
训练时随机采 $\alpha$ 当条件喂给网络，推理时 $\alpha$ 任意指定，1-NFE 属性不掉。

## 跟 Song 2021 对照

| 维度 | Song 2021 | Drifting |
|---|---|---|
| 什么在演化 | 样本沿时间 $t$ | pushforward 分布沿训练迭代 $i$ |
| Drift 定义 | $f(x,t) - g^2\nabla\log p_t$ | $V_{p,q}$ = attraction − repulsion |
| 依赖的目标信息 | Score（点估计） | 两分布的成对交互（kernel） |
| 训练目标 | DSM | Fixed-point MSE + stop-grad |
| 推理 | 反向 SDE / PF-ODE，多步 | 一次前向 $f_\theta(\epsilon)$，1-NFE |
| 均衡条件 | $s_\theta \to \nabla\log p_t$（点态） | $V = 0$（分布态） |

共同点：都以"漂移场"为核心对象，两者都是把 $q$ 推向 $p_{\text{data}}$，
只不过 Song 的 $q_t$ 是 forward SDE 制造出来的 corrupted 分布，这里的 $q$
是网络当前 pushforward。

**本质偏离**：
- Song 的 drift 是**采样器**的一部分（inference dynamics）；
- Drifting 的 drift 是**优化器**的一部分（training dynamics）。
- Song 学"给定 $t$ 时刻分布，score 是什么"，是**刻画路径**；
  Drifting 学"如何一步把噪声推到数据"，是**跳过路径**。

一个可能的桥：把 Song 的 PF-ODE 从 $t=T$ 到 $t=0$ 整条积出来定义成一个
$f_\theta(\epsilon)$，那 Drifting 就是**直接学这个映射**而不显式跑中间 $t$。
—— 这个视角能不能形式化，读附录时重点看。也可能是我明天问导师的问题。

## 我这次没搞懂的 / 存疑

- Prop. 3.1 逆命题：Appx C.1 那个 heuristic 具体是几条 bilinear 约束堆出来的？
  跟 kernel MMD (Gretton 2012) 的 characteristic kernel 假设是不是等价？
  —— **这是文章最脆的一环，我得搞清楚**。
- Alg. 1 里 $y^-$ 直接 reuse 同 batch 的生成样本 $x$。SGD 早期 $q$ 离 $p$ 很远，
  $V$ 幅度会不会爆？有没有 warmup？没看到明确的。
- Kernel 用 $\exp(-\|x-y\|/\tau)$ 而不是 $\exp(-\|x-y\|^2/\tau)$：为什么 $\ell_1$ 型更稳？
  和 InfoNCE 常用的 cosine similarity 有没有 head-to-head？
- $\tau$ 怎么选？跟 batch size $N_{\text{neg}}, N_{\text{pos}}$ 的 scaling？
- $\phi$ 换成 DINOv2 / CLIP / MAE 是不是也可以？敏感性分析节在哪一节？没找到。
- "extra softmax over $\{x\}$ within a batch"（§3.3 kernel 段落）—— InfoNCE 平时只在
  负样本维度做一次 softmax，这里第二个 softmax 的归一化维度我没读清。

## 可能能顺下去的方向

（草记，checkpoint / HKPFS proposal 素材候选）

- 路径 vs 端到端：drifting 完全跳路径。能不能加一条 coarse-drift + fine-drift
  的中间监督，得到 few-step 而非严格 1-step 的质量-速度 tradeoff？
- Drifting field 的替代形式：现在的 mean-shift + softmax kernel 只是一个朴素实例。
  Wasserstein gradient flow / Stein variational gradient descent (Liu & Wang 2016)
  的 drift 拿来做替代如何？
- **Identifiability**：现在只有 heuristic。能否在 characteristic kernel + universal
  approximator 假设下证一个 clean 的 "$V=0 \Rightarrow q=p$"？这是导师风格的活，
  我估计他会推我做这个方向 —— 明天开会先探一下口风。
- 训练时演化 vs 推理时演化的形式关系：能不能把 Drifting 写成 PF-ODE 在训练动态下的
  极限？如果能，就把这条线跟 Song 2021 缝上，也顺便串起 flow matching。
- Off-policy 负样本：作者用当前 batch 的 $x$，纯 on-policy。加一个 replay buffer 或
  slow-moving EMA 生成器，负样本更多样，能不能加速收敛？—— MoCo 的 memory bank 思路
  （He et al. 2020，正是这篇作者），迁移过来非常自然。
- Robotic control (§5.3)：Diffusion policy 已经是 robot learning 标配了。单步 drifting
  policy 如果 task success 不掉，就有很直接的落地。可做一个小 head-to-head。

## 待办

- [ ] 明天先重读 §3.3 的 kernel 归一化细节，把两次 softmax 的维度画清；
- [ ] Appx C.1 identifiability heuristic 逐行推一遍；
- [ ] Alg. 1 抄下来对着代码复盘（如果作者放了 code）；
- [ ] 跟导师碰的时候问三件：identifiability 有没有做过、feature encoder 敏感性、
      能不能把这个 framing 往 form of PF-ODE limit 推。

---

**状态**：主文读完一遍，附录 + Alg. 未细看
**开始**：7.19
**下一步**：Appx C.1 + §3.3 kernel 细节
