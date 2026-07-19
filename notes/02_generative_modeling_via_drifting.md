# 02 · Generative Modeling via Drifting

> Deng, M., Li, H., Li, T., Du, Y., & He, K. (2026).
> *Generative Modeling via Drifting*.
> arXiv: https://arxiv.org/abs/2602.04770 (v2). 28 pages. CC-BY 4.0.
> 作者单位：MIT (Deng / H. Li / T. Li / K. He) + Harvard (Du)。
> **读完 01_song_sde 之后再读**，确保 SDE / drift term 语言已掌握。

## 0 · 为什么读这篇

- 导师指定 → 顺着这条思路推进最快能对齐他的科研主线。
- "drifting" 字面就是 SDE 里的 drift term $f(x,t)$，跟 flow matching
  的 velocity field $v(x,t)$ 是同一个东西的两个视角。这篇很可能给出**统一
  两者的一个具体 framing**。
- 关键卖点：**one-step 生成** —— ImageNet 256×256 单步 FID 1.54 (latent) /
  1.61 (pixel)，直接对标 diffusion / flow 多步采样。如果 claim 成立，这是
  近期 "one-NFE generator" 路线里数字最好的之一，很值得逐节拆。

## 1 · 一句话总结

不再让样本在**推理时**沿 SDE / ODE 一步步演化到数据分布，而是让 pushforward
分布 $q = f_\#p_\epsilon$ 在**训练时**沿一个 "drifting field" $V_{p,q}(x)$ 演化
到 $p_{\mathrm{data}}$；把 $V=0 \Leftrightarrow p=q$ 的 anti-symmetric 场用一个
attraction (真实样本) − repulsion (生成样本) 的 mean-shift 形式实例化，再配
stop-gradient 的 MSE fixed-point loss，训练完的 $f_\theta$ 就是一步生成器。

## 2 · 关键公式

### 2.1 Pushforward 与训练时演化 (§3.1)

- 生成器 $f:\mathbb{R}^C \to \mathbb{R}^D$，$x=f(\epsilon)$，$\epsilon\sim p_\epsilon$。
- 输出分布记 $q = f_\# p_\epsilon$（$f$ 把 $p_\epsilon$ 推前得到的分布）。
- SGD 迭代给出模型序列 $\{f_i\}$ 和分布序列 $\{q_i\}$；样本在训练中被"漂移"：
  $x_{i+1} = x_i + \Delta x_i$，$\Delta x_i := f_{i+1}(\epsilon) - f_i(\epsilon)$。
- **核心视角切换**：SDE 视角里 drift 让样本在**时间**上演化；这里 drift 让
  样本随**训练迭代**演化。

### 2.2 Drifting field 与反对称性 (§3.2)

Drifting field $V_{p,q}:\mathbb{R}^d\to\mathbb{R}^d$ 满足更新方程

$$
x_{i+1} = x_i + V_{p,q_i}(x_i)\tag{2}
$$

**Prop. 3.1（反对称 ⇒ 均衡）**：若 $V_{p,q}(x)=-V_{q,p}(x)\ \forall x$，则
$q=p \Rightarrow V_{p,q}(x)=0$。逆命题一般不成立，作者只在 §3.3 的核形式下
给出 "$V\approx 0 \Rightarrow q\approx p$" 的充分条件（Appx C.1）。

### 2.3 Fixed-point 训练目标 (§3.2)

最优参数 $\hat\theta$ 满足 fixed point：$f_{\hat\theta}(\epsilon) = f_{\hat\theta}(\epsilon) + V_{p,q_{\hat\theta}}(f_{\hat\theta}(\epsilon))$。
迭代式转成 loss：

$$
\mathcal{L} = \mathbb{E}_\epsilon\Bigl\|\,\underbrace{f_\theta(\epsilon)}_{\text{prediction}}
- \operatorname{stopgrad}\bigl(\underbrace{f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))}_{\text{frozen target}}\bigr)\Bigr\|^2 \tag{6}
$$

- 数值上等于 $\mathbb{E}_\epsilon\|V(f(\epsilon))\|^2$，但因为 stop-grad，梯度**不穿过 $V$**
  （$V$ 依赖 $q_\theta$，穿过一个分布不好办）。
- 血缘：stop-grad + 自蒸馏 MSE 形式跟 SimSiam (Chen & He 2021) 和 CT
  (Song & Dhariwal 2023) 是同一套技巧。

### 2.4 核化 drifting field (§3.3)

$$
V_{p,q}(x) = \mathbb{E}_{y^+\sim p}\,\mathbb{E}_{y^-\sim q}\bigl[K(x,y^+,y^-)\bigr]\tag{7}
$$

作者选用 mean-shift (Cheng 1995) 型实例：

$$
V^+_p(x) = \tfrac{1}{Z_p}\mathbb{E}_p[k(x,y^+)(y^+-x)],\quad
V^-_q(x) = \tfrac{1}{Z_q}\mathbb{E}_q[k(x,y^-)(y^--x)] \tag{8}
$$

$$
V_{p,q}(x) := V^+_p(x) - V^-_q(x)
= \tfrac{1}{Z_pZ_q}\mathbb{E}_{p,q}[k(x,y^+)k(x,y^-)(y^+-y^-)]\tag{10,11}
$$

- 直觉：$x$ 被真实样本 $y^+$ **吸引**，被生成样本 $y^-$ **排斥**。
- Kernel：$k(x,y) = \exp(-\tfrac1\tau \|x-y\|)$，用 **softmax** 数值实现，
  logits $= -\tfrac1\tau \|x-y\|$ 沿 $y$ 归一化；额外再在 batch 的 $\{x\}$ 上做
  一遍 softmax 归一化（经验上提升），不破坏反对称性。
- 与 InfoNCE：normalized $\tilde k$ 的 softmax 形式和 InfoNCE (Oord 2018) 同构，
  可以理解成"对比学习 + mean-shift"的杂交。

### 2.5 特征空间 drifting (§3.4)

高维图像不能在像素空间直接算 kernel。改到特征空间 $\phi(\cdot)$（预训练的
自监督 image encoder，如 MoCo 家族）：

$$
\mathbb{E}\Bigl\|\phi(x) - \operatorname{stopgrad}\bigl(\phi(x) + V(\phi(x))\bigr)\Bigr\|^2 \tag{13}
$$

多尺度多位置版本：对 ResNet 各 stage 的 $\phi_j$ 分别做 loss 再求和 (Eq. 14)。
$\phi$ 只在**训练时**用，推理仍然是一步 $f_\theta$。

- **与 perceptual loss 的区别**：perceptual loss 是 $\|\phi(x)-\phi(x_{\text{target}})\|^2$，
  需要配对目标；这里的回归目标是 $\phi(x)+V(\phi(x))$，**不需要配对**，
  匹配的是 pushforward 分布 $\phi_\# q \to \phi_\# p$。
- **与 latent generation 正交**：$f_\theta$ 可以是 pixel-space 或 SD-VAE latent-space
  的生成器；$\phi$ 是**另一个**空间的编码器；两者独立选。

### 2.6 Classifier-Free Guidance (§3.5)

用 mixture 负样本分布替换 $q$：

$$
\tilde q(\cdot|c) = (1-\gamma)\,q_\theta(\cdot|c) + \gamma\,p_{\text{data}}(\cdot|\emptyset) \tag{15}
$$

要求 $\tilde q(\cdot|c)=p_{\text{data}}(\cdot|c)$ 推出

$$
q_\theta(\cdot|c) = \alpha\,p_{\text{data}}(\cdot|c) - (\alpha-1)\,p_{\text{data}}(\cdot|\emptyset),
\quad \alpha = \tfrac{1}{1-\gamma}\ge 1 \tag{16}
$$

- 精神上和标准 CFG (Ho & Salimans 2022) 一致：conditional 减去 unconditional。
- 关键差别：**CFG 是训练时行为** —— 负样本抽样 + $\alpha$ 作为条件喂给网络；
  推理时 1-NFE 属性完全保留。作者的具体做法是 CFG-conditioning (Geng 2025b)：
  训练时随机采样 $\alpha$ 作为条件；推理时 $\alpha$ 可任意指定不需重训。

## 3 · 主要贡献

1. **新 paradigm**：把生成建模从"推理时演化样本"转成"训练时演化 pushforward
   分布"，天然是 one-step (1-NFE) 生成器 —— 无需 diffusion / flow 的多步采样。
2. **Anti-symmetric drifting field**：给出一个简单充分条件 (Prop. 3.1) 保证
   $q=p \Rightarrow V=0$；并用 mean-shift + attraction/repulsion 给出可计算的
   核化实例 (Eq. 8-11)。
3. **训练目标**：stop-gradient fixed-point MSE (Eq. 6)，绕开了对 $V$ 内部 $q$
   的反向传播；实现上就是一行 `mse(x, stopgrad(x+V))`。
4. **Feature-space drifting**：把 loss 挪到自监督 encoder 的特征空间，多尺度
   多位置聚合；解决高维直接用 kernel 的失败模式。
5. **训练时 CFG**：把 CFG 表达为负样本分布的 mixture，保持 1-NFE。
6. **实证结果**：ImageNet 256×256 单步生成 FID **1.54** (latent) / **1.61**
   (pixel space)，作者宣称是新的 one-step SOTA；另外做了 toy 2D 演化可视化
   (§5.1) 和 robotic control policy 生成 (§5.3) 的迁移实验。

## 4 · 跟 Song 2021 的关系

| 维度 | Song 2021 (SDE) | Drifting Model |
|---|---|---|
| 什么在演化 | **样本** 沿时间 $t$ 演化 | **pushforward 分布** 沿训练迭代 $i$ 演化 |
| Drift 定义 | $f(x,t) - g^2 \nabla_x\log p_t(x)$ | $V_{p,q}(x)$ = 真实样本吸引 − 生成样本排斥 |
| 依赖的目标信息 | Score $\nabla_x\log p_t$（点估计） | 两个分布 $p,q$ 的成对交互（kernel-based） |
| 训练目标 | Denoising score matching (回归 Gaussian 转移核的 score) | Fixed-point MSE with stop-grad |
| 推理 | Reverse SDE / probability flow ODE，多步 | 单次前向 $f_\theta(\epsilon)$，1-NFE |
| 均衡条件 | $s_\theta \to \nabla \log p_t$（点态） | $V=0$（分布匹配的分布态） |

**共同点**：都以一个"漂移场"作为核心对象；反向 SDE 的 drift 与 probability
flow ODE 的 drift 都在做"把 $q_t$ 推向 $p_{\text{data}}$"这件事，只是那里的
$q_t$ 是 forward SDE 制造的 corrupted 分布，这里的 $q$ 是网络当前的 pushforward。

**本质偏离**：
- Song 的 drift 是**采样器**的一部分（inference-time dynamics）；
- Drifting model 的 drift 是**优化器**的一部分（training-time dynamics）。
- Song 学的是"给定 $t$ 时刻的分布，得分是多少"；Drifting model 学的是
  "怎样一步把噪声推到数据"。**前者刻画路径，后者跳过路径**。

**Probability flow ODE 的可能桥梁**：如果把 Song 的 probability flow ODE 从
$t=T$ 到 $t=0$ 的整条积分定义为一个 $f_\theta(\epsilon)$，那这个 $f_\theta$ 是一个
multi-step 的一次映射；drifting model 相当于**直接学这个映射**而不显式经过
中间 $t$。这个视角能不能形式化 —— **读到附录时重点看**。

## 5 · 我不理解的点

（读的过程中随时记）

- [ ] Prop. 3.1 逆命题不成立，但作者在附录 C.1 给了"kernel + 温和 non-degeneracy
      假设下 $V\approx0 \Rightarrow q\approx p$" 的识别性 heuristic。这个 heuristic
      具体是几个 bilinear 约束堆出来的？跟 kernel MMD (Gretton 2012) 的
      characteristic kernel 假设是否等价？
- [ ] 训练时 $y^-$ 直接 reuse 同 batch 的生成样本 $x$（Alg. 1），这在 SGD 早期
      $q$ 离 $p$ 很远时会不会造成 $V$ 幅度过大 → loss 爆炸？作者有没有 warmup？
- [ ] Kernel 用 $\exp(-\|x-y\|/\tau)$ 而不是常见的 $\exp(-\|x-y\|^2/\tau)$（RBF），
      $\ell_1$ 型 exp 在实践里为什么更稳？跟 InfoNCE 常用的 cosine similarity 有没有比过？
- [ ] 温度 $\tau$ 怎么选？跟 batch 大小 $N_{\text{neg}}, N_{\text{pos}}$ 的依赖关系？
- [ ] Feature encoder $\phi$ 换成 DINOv2 / CLIP / MAE 之后 FID 会如何变化？
      作者对 $\phi$ 的敏感性分析在哪一节？
- [ ] "Extra softmax over $\{x\}$ within a batch"（§3.3 kernel 段落）是什么形式？
      InfoNCE 里通常只在负样本维度做 softmax，这里加的第二个 softmax 到底在归一化什么？

## 6 · 跟导师后续工作的连接线索

（哪些点像是可以往下扩的方向？记下来 —— 这是 checkpoint / HKPFS proposal 的候选素材）

- **路径 vs 端到端**：drifting model 跳过了显式路径。可否引入一个可控的中间监督
  （e.g. 学两条 drifting field：一条粗、一条细），得到 few-step 而非严格 1-step
  的可控质量-速度 tradeoff？
- **Drifting field 的选择**：现在的 mean-shift + softmax kernel 是一种朴素实例。
  能否借鉴 Wasserstein gradient flow / Stein variational gradient descent
  (Liu & Wang 2016) 里的 drift 形式，用其他 characteristic kernel？
- **$V=0 \Rightarrow q=p$ 的充分条件**：识别性目前只有 heuristic。能否在
  更小的假设集 (characteristic kernel, universal approximator $f_\theta$) 下证一个
  clean 的 identifiability theorem？—— 这是**导师可能希望我做**的方向，
  因为他风格偏理论；**读完再跟他确认**。
- **训练时演化 vs 推理时演化的形式关系**：能否把 drifting model 严格写成
  probability flow ODE 在训练动态下的极限？如果能，就把这条线跟 Song 2021 缝上，
  也顺带给出 flow matching / rectified flow 的另一个视角。
- **Off-policy 负样本**：作者用当前 batch 的 $x$ 当负样本，是纯 on-policy。
  如果引入一个 replay buffer 或 slow-moving EMA 生成器，负样本更多样化，
  收敛能不能加速？（模仿 MoCo 的 memory bank，He et al. 2020，正是本文作者。）
- **Robotic control 那节 (§5.3)**：Diffusion policy 现在是 robot learning 的标配，
  单步 drifting policy 如果 FID / task success 都不吃亏，那有非常直接的落地价值。
  可以做**跟 diffusion policy (Chi et al.) 头对头 benchmark** 当一个小项目。

---

**状态**：未开始
**开始日期**：
**完成日期**：
