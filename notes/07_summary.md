# 07 · 前 6 篇的统一总结

- 日期：7.21
- 覆盖：01 NCSN · 02 DDPM · 03 Flow Matching · 04 CFG · 05 Song SDE · 06 Drifting

## 0 · 时间轴与谱系

```
2019   NCSN  (score matching + Langevin, VE)
        │
2020   DDPM  (Markov chain, ε-prediction, VP)
        │
2021   Song SDE  (统一 VE / VP, PF-ODE)
        │
2022   Flow Matching  (velocity, OT 路径)
        │
2022   CFG  (无分类器引导)
        │
2026   Drifting  (training-time pushforward 演化, 1-NFE)
```

三条演化线：
- 从**离散**（DDPM / NCSN）到**连续**（Song SDE）；
- 从 SDE 到 **ODE / velocity**（PF-ODE → Flow Matching）；
- 从 **inference-time 演化样本**（前 5 篇）到 **training-time 演化分布**（Drifting）。

## 1 · 核心对象的三种化身

同一条"把噪声推向数据"的信息，被三种代理量刻画：

| 代理量 | 定义 | 出现在 |
|---|---|---|
| Score $s(x,t)$ | $\nabla_x \log p_t(x)$ | NCSN, DDPM, Song SDE |
| Velocity $v(x,t)$ | $f(x,t) - \tfrac12 g(t)^2 s(x,t)$ | PF-ODE, Flow Matching |
| Drifting field $V_{p,q}(x)$ | attraction($p$) − repulsion($q$) | Drifting |

关键换算：

$$
\varepsilon_\theta(x_t, t) = -\sqrt{1-\bar\alpha_t}\,s_\theta(x_t, t)\qquad(\text{DDPM} \leftrightarrow \text{Score})
$$

$$
v(x, t) = f(x, t) - \tfrac12 g(t)^2\,s(x, t)\qquad(\text{SDE} \leftrightarrow \text{PF-ODE})
$$

$$
V_{p,q}(x) = \mathbb{E}_{y^+\sim p, y^-\sim q}\bigl[K(x, y^+, y^-)\bigr]\qquad(\text{Drifting})
$$

## 2 · 训练目标对比

| 论文 | 训练目标 | 回归物 | 参数化 |
|---|---|---|---|
| NCSN | multi-scale DSM | $\nabla\log q_\sigma(\tilde x\mid x)$ | $s_\theta(x, \sigma)$ |
| DDPM | $L_{\text{simple}}$（加权 DSM） | $\varepsilon$（等价 score） | $\varepsilon_\theta(x_t, t)$ |
| Song SDE | 连续时间 DSM，$\lambda(t)=g(t)^2$ | $\nabla\log p_{0t}(x_t\mid x_0)$ | $s_\theta(x, t)$ |
| Flow Matching | (Conditional) Flow Matching | 条件 velocity $u_t(x\mid z)$ | $v_\theta(x, t)$ |
| CFG | 与 DDPM 相同 + 训练时随机置 $y=\emptyset$ | $\varepsilon$ | $\varepsilon_\theta(x_t, t, y)$ |
| Drifting | Fixed-point MSE + stop-grad | $x + V(x)$ 的自身 | $f_\theta(\epsilon)$ |

共同点：**MSE 回归一个由 forward 过程解析给出的目标**（Drifting 用 $V$ 从分布对比造出目标）。

## 3 · 采样对比

| 论文 | 采样方式 | 步数量级 | 随机 / 确定 |
|---|---|---|---|
| NCSN | Annealed Langevin | $L\times T$（$L=10$，$T=100$） | 随机 |
| DDPM | Ancestral sampling | $\sim 1000$ | 随机 |
| Song SDE | Reverse SDE (PC) / PF-ODE | 100–1000 / 50–200 | 两者兼具 |
| Flow Matching | ODE solver（RK4/Heun/Dopri5） | 5–100 | 确定 |
| CFG | DDPM/DDIM + 每步 2 次 NFE | $\sim 2\times$ DDPM | 随机 |
| Drifting | 单次前向 $f_\theta(\epsilon)$ | **1** | 确定 |

趋势：$\text{NCSN/DDPM}\to \text{ODE}\to \text{FM}\to \text{Drifting}$，采样步数递减。

## 4 · Forward 过程 / 概率路径

| 论文 | 参数化 | 端点 |
|---|---|---|
| NCSN | 加性高斯，$\sigma_1 > \dots > \sigma_L$（VE 型） | $x + \sigma\varepsilon$ |
| DDPM | Markov，$\bar\alpha_t = \prod(1-\beta_s)$（VP 型） | $\sqrt{\bar\alpha_t}x + \sqrt{1-\bar\alpha_t}\varepsilon$ |
| Song SDE | SDE：$dx = f(x,t)dt + g(t)dW$；VE / VP / sub-VP | 由 SDE 决定 |
| Flow Matching | 任意条件高斯路径 $p_t(x\mid x_1)$；主推 OT 直线 | $x_t = t x_1 + (1-t) x_0$ |
| Drifting | **无显式路径**；只有分布 $q_i \to p_{\text{data}}$ 沿训练迭代 | — |

Song SDE 提供的关键身份：VP-SDE 的 Euler-Maruyama 离散 = DDPM；VE-SDE 的对应 = NCSN。Flow Matching 的 OT 路径把 forward 从 diffusion 解耦，允许直线插值。

## 5 · Score / velocity / drift 的相互关系

以 VP 系数 $\mu_t = \alpha_t x_1,\ \sigma_t^2 = 1 - \alpha_t^2$（$\alpha_t = \sqrt{\bar\alpha_t}$）为例，同一条概率路径下四种量的换算：

$$
x_t = \alpha_t x_1 + \sigma_t \varepsilon
$$

$$
\varepsilon = -\sigma_t\,\nabla_{x_t}\log p_t(x_t)
= -\sigma_t\,s(x_t, t)
$$

$$
v_t(x_t\mid x_1) = \alpha_t' x_1 + \sigma_t'\,\varepsilon
= \alpha_t' \Bigl(\tfrac{x_t - \sigma_t \varepsilon}{\alpha_t}\Bigr) + \sigma_t'\,\varepsilon
$$

化简：

$$
v_t(x_t\mid x_1) = \tfrac{\alpha_t'}{\alpha_t}\,x_t + \Bigl(\sigma_t' - \tfrac{\alpha_t'\sigma_t}{\alpha_t}\Bigr)\varepsilon
$$

选 $\alpha_t = e^{-\tfrac12 \int_0^t \beta(s)ds}$ 使 $\tfrac{\alpha_t'}{\alpha_t} = -\tfrac12 \beta(t) = f(x_t, t)/x_t$，代入可以还原到 PF-ODE 的 $v = f - \tfrac12 g^2 s$。

**结论**：$\varepsilon$、score、velocity 在给定概率路径下是等价的三种参数化，只差一组线性变换；Drifting 的 $V$ 处于另一层——它不属于任一具体路径，而是**分布之间的一次跳跃**。

## 6 · 条件生成的统一表达

给定条件 $y$，目标分布：

$$
\tilde p_t(x\mid y) \propto p_t(x) \cdot p_t(y\mid x)^s
$$

- **Classifier Guidance**（Dhariwal-Nichol 2021）：显式训一个 $p_\phi(y\mid x_t)$，加到 score 上。
- **CFG**（04）：训一个共享 $\varepsilon_\theta(x_t, t, y)$，训练时以 $p_{\text{uncond}}$ 概率把 $y$ 换成 $\emptyset$；推理时 $\tilde\varepsilon = \varepsilon_\emptyset + s(\varepsilon_y - \varepsilon_\emptyset)$。
- **Drifting 的训练时 CFG**（06 §3.5）：把 $\alpha$ 作为条件网络输入，训练时随机采 $\alpha$；推理仍单次前向。

三者共享的形式：

$$
\text{guided score} = (1-s)\,\nabla\log p(x) + s\,\nabla\log p(x\mid y)
$$

差异仅在 $\nabla \log p(y\mid x)$ 的**获取方式**（外部分类器 / 隐式差分 / 训练时分布 mixture）。

## 7 · 一步 vs 多步

| 层级 | 代表 | 每次生成 NFE | 是否显式经过中间 $t$ |
|---|---|---|---|
| 多步 SDE | DDPM, NCSN, Song SDE | 数百—上千 | 是 |
| 多步 ODE | PF-ODE, Flow Matching | 5–100 | 是 |
| 一步 (distillation) | Consistency Models, Progressive Distillation | 1–4 | 隐式 |
| 一步 (直接学) | Drifting | **1** | 否 |

Drifting 与 Consistency Models 的关键差异：Consistency 蒸馏一个已有 PF-ODE 的解算子，仍继承一条明确路径；Drifting 完全不参照路径，只用分布对比作监督。

## 8 · 分布匹配的两种时间轴

统摄性视角：所有 6 篇都在把"当前分布 $q$"匹配"数据分布 $p$"。差别在于"当前分布"是**如何被参数化**、**时间**代表什么：

| 论文 | "当前分布" | 时间轴 |
|---|---|---|
| NCSN / DDPM / Song SDE | 加噪核 $q_\sigma$ 或 $q(x_t\mid x_0)$ | 物理噪声尺度 |
| Flow Matching | 条件路径 $p_t(x\mid z)$ | ODE 时间 $t\in[0,1]$ |
| CFG | 加噪条件核 $q(x_t\mid x_0, y)$ | 与 DDPM 同 |
| Drifting | 生成器 pushforward $q_i = f_{\theta_i\#}\,p_\epsilon$ | 训练迭代 $i$ |

一句话：Song 2021 之前的所有工作把 $q$ 参数化为 forward 过程的**边缘**，Drifting 把 $q$ 参数化为**生成器输出**。前者用 $t$（物理时间）平滑 $q$，后者用 $\theta$（参数）演化 $q$。

## 9 · 关键身份/等式汇总

$$
s_\theta(x_t, t) = -\varepsilon_\theta(x_t, t)/\sqrt{1-\bar\alpha_t} \tag{DDPM ↔ Score}
$$

$$
v(x, t) = f(x, t) - \tfrac12 g(t)^2\,s(x, t) \tag{SDE ↔ PF-ODE}
$$

$$
u_t(x\mid x_1) = \tfrac{\sigma_t'}{\sigma_t}(x - \mu_t) + \mu_t' \tag{FM conditional velocity}
$$

$$
V_{p,q}(x) = V^+_p(x) - V^-_q(x),\quad V^\pm(x) \propto \mathbb{E}[k(x, y)(y - x)] \tag{Drifting mean-shift}
$$

$$
\tilde\varepsilon = \varepsilon_\emptyset + s\,(\varepsilon_y - \varepsilon_\emptyset) \tag{CFG}
$$

## 10 · 落到 Drifting proposal 的挂钩

- **Identifiability**：Drifting 的 $V\equiv 0 \Rightarrow q=p$ 目前只有 heuristic；类比 kernel MMD 的 characteristic kernel 假设可给出严格证明的候选路径。
- **训练动态的连续极限**：把 SGD 迭代抬升为 $\partial_\tau q + \nabla\cdot(q V) = 0$，可能揭示 Drifting 是某泛函的 $W_2$ gradient flow；与 Song 2021 的 PF-ODE 形成对偶（物理时间 vs 训练时间）。
- **Few-step 拓展**：把 1-NFE Drifting 拓展成 $K$ 层级 $V^{(1)},\dots,V^{(K)}$，介于 Flow Matching 与 Drifting 之间的连续插值。
- **Path-free vs path-based 的形式关系**：能否把 PF-ODE 从 $t=T$ 到 $t=0$ 的积分算子写成 $f_\theta$，从而把 Drifting 表达为 Flow Matching 的"整段积分极限"？

---

状态：已完
