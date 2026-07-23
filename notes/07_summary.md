# 07 · 关于前 6 篇论文的阅读小结

- 日期：7.22
- 覆盖：01 NCSN · 02 DDPM · 03 Flow Matching · 04 CFG · 05 Song SDE · 06 Drifting

## 阅读结论

- **共同主线**：六篇都在回答怎样把简单分布变成数据分布，但选择了不同的监督量与演化轴。
- **路径内参数化**：NCSN、DDPM、Song SDE 学 score / noise；Flow Matching 直接学 velocity。给定同一概率路径时，它们可通过条件期望和线性变换互换。
- **路径外控制**：CFG 不定义新的生成路径，而是在已有 diffusion score 上做 conditional-unconditional 外推。
- **训练时间演化**：Drifting 不沿样本时间积分，而让生成器的 pushforward 分布随优化迭代移动；它与 SDE 中的 drift 同名但不是同一个数学对象。

## 0 · 时间轴与谱系

```text
2019  NCSN ── VE / score / Langevin ──┐
                                      ├─ 2021 Song SDE ── PF-ODE ── 2022 Flow Matching
2020  DDPM ── VP / epsilon / Markov ──┘                    │
          └─ conditional + unconditional score ── 2021/2022 CFG

2026  Drifting ── training-time pushforward evolution / native 1-NFE
```

四条关系线：
- 从**离散**（DDPM / NCSN）到**连续**（Song SDE）；
- 从 SDE 到 **ODE / velocity**（PF-ODE → Flow Matching）；
- CFG 横向修改条件 score，不改变底层 diffusion 时间轴；
- 从 **inference-time 演化样本**（NCSN / DDPM / SDE / FM）到 **training-time 演化 pushforward 分布**（Drifting）。

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
| Song SDE | 连续时间 DSM，$\lambda(t)>0$ | $\nabla\log p_{0t}(x_t\mid x_0)$ | $s_\theta(x, t)$ |
| Flow Matching | (Conditional) Flow Matching | 条件 velocity $u_t(x\mid z)$ | $v_\theta(x, t)$ |
| CFG | 与 DDPM 相同 + 训练时随机置 $y=\emptyset$ | $\varepsilon$ | $\varepsilon_\theta(x_t, t, y)$ |
| Drifting | Fixed-point MSE + stop-grad | $x + V(x)$ 的自身 | $f_\theta(\epsilon)$ |

共同点：大多可写成 MSE，但监督来源不同。NCSN、DDPM、Song SDE 的目标来自加噪核；FM 来自选定条件路径的导数；Drifting 则从当前生成分布与数据分布的成对比较构造冻结目标。

## 3 · 采样对比

| 论文 | 采样方式 | 步数量级 | 随机 / 确定 |
|---|---|---|---|
| NCSN | Annealed Langevin | $L\times T$（$L=10$，$T=100$） | 随机 |
| DDPM | Ancestral sampling | $\sim 1000$ | 随机 |
| Song SDE | Reverse SDE (PC) / PF-ODE | 100–1000 / 50–200 | 两者兼具 |
| Flow Matching | ODE solver（RK4/Heun/Dopri5） | 原论文约 122–193 NFE | 确定 |
| CFG | 继承 DDPM/DDIM；每步 cond + uncond | 基础 sampler 的 2 倍网络前向 | 取决于基础 sampler |
| Drifting | 单次前向 $f_\theta(\epsilon)$ | **1** | 确定 |

这里不能简单读成“论文越新、步数必然越少”：FM 原论文仍是百级 NFE；Drifting 的 1-NFE 则以更重的训练端分布估计为代价。NFE 还必须区分一次 solver step 与一次网络前向。

## 4 · Forward 过程 / 概率路径

| 论文 | 参数化 | 端点 |
|---|---|---|
| NCSN | 加性高斯，$\sigma_1 > \dots > \sigma_L$（VE 型） | $x + \sigma\varepsilon$ |
| DDPM | Markov，$\bar\alpha_t = \prod(1-\beta_s)$（VP 型） | $\sqrt{\bar\alpha_t}x + \sqrt{1-\bar\alpha_t}\varepsilon$ |
| Song SDE | SDE：$dx = f(x,t)dt + g(t)dW$；VE / VP / sub-VP | 由 SDE 决定 |
| Flow Matching | 任意条件高斯路径 $p_t(x\mid x_1)$；主推 OT 直线 | $x_t = t x_1 + (1-t) x_0$ |
| CFG | 不新增 forward path，继承所用 diffusion 模型 | 与基础模型相同 |
| Drifting | **无显式路径**；只有分布 $q_i \to p_{\text{data}}$ 沿训练迭代 | — |

Song SDE 提供的关键身份：VP-SDE 的 Euler-Maruyama 离散 = DDPM；VE-SDE 的对应 = NCSN。Flow Matching 的 OT 路径把 forward 从 diffusion 解耦，允许直线插值。

## 5 · Score / velocity / drift 的相互关系

先统一方向：令 $t:0\to T$ 表示从数据到噪声，VP 边缘写成

$$
x_t = a_t x_0 + b_t\varepsilon,
\qquad
a_t^2+b_t^2=1,
\qquad
\varepsilon\sim\mathcal{N}(0,I).
$$

对单个训练对，条件 score target 为 $-\varepsilon/b_t$；边缘 score 则是它在给定 $x_t$ 后的条件期望（Tweedie identity）：

$$
\mathbb{E}[\varepsilon\mid x_t]
=-b_t\,\nabla_{x_t}\log p_t(x_t)
=-b_t\,s(x_t,t).
$$

因此“$\varepsilon=-b_t s$”不是对每次采到的噪声逐点成立，而是 MSE 最优预测器满足的条件期望关系。

同一条件路径的 velocity target 是

$$
u_t(x_t\mid x_0)=a_t'x_0+b_t'\varepsilon.
$$

边缘 velocity 是 $v_t(x)=\mathbb{E}[u_t\mid x_t=x]$。消去 $x_0$ 并代入上面的条件期望：

$$
v_t(x)
=\frac{a_t'}{a_t}x
-b_t\left(b_t'-\frac{a_t'b_t}{a_t}\right)s(x,t).
$$

VP-SDE 中 $a_t=\exp[-\tfrac12\int_0^t\beta(s)ds]$、$b_t^2=1-a_t^2$，上式化为

$$
v_t(x)=-\frac12\beta(t)x-\frac12\beta(t)s(x,t)
=f(x,t)-\frac12g(t)^2s(x,t),
$$

正是 data-to-noise 方向的 PF-ODE。Flow Matching 常用 noise-to-data 方向，只需反转时间与 velocity 符号。

**结论**：noise prediction、score 与 marginal velocity 在给定概率路径下可互换，但转换经过条件期望，并非任意逐样本恒等式。Drifting 的 $V_{p,q}$ 处于另一层：它描述训练迭代中的分布更新，不是该路径上的 velocity。

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

三者都有“conditional 相对 unconditional 的外推”结构，但作用对象不同：classifier guidance 直接给 score 加梯度，CFG 对两个去噪预测做差，Drifting 用真实无条件样本改变训练时负样本分布。

## 7 · 一步 vs 多步

| 层级 | 代表 | 每次生成 NFE | 是否显式经过中间 $t$ |
|---|---|---|---|
| 多步 SDE | DDPM, NCSN, Song SDE | 数百—上千 | 是 |
| 多步 ODE | PF-ODE, Flow Matching | 原论文通常几十到数百 NFE | 是 |
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

一句话：NCSN、DDPM、Song SDE 与 FM 显式指定由 $t$ 索引的中间分布；Drifting 直接把“当前分布”设为生成器 pushforward $q_\theta$，由优化更新 $\theta$ 来演化它。CFG 是控制层，不单独定义这两种时间轴。

## 9 · 关键身份/等式汇总

**DDPM noise prediction 与 score：**

$$
s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}
$$

**SDE 与 probability flow ODE：**

$$
v(x, t) = f(x, t) - \frac12 g(t)^2\,s(x, t)
$$

**FM Gaussian conditional velocity：**

$$
u_t(x\mid x_1) = \frac{\sigma_t'}{\sigma_t}(x - \mu_t) + \mu_t'
$$

**Drifting mean-shift field：**

$$
V_{p,q}(x) = V^+_p(x) - V^-_q(x),
\qquad
V^\pm(x) \propto \mathbb{E}[k(x, y)(y - x)]
$$

**Classifier-Free Guidance：**

$$
\tilde\varepsilon = \varepsilon_\emptyset + s\,(\varepsilon_y - \varepsilon_\emptyset)
$$

## 10 · 六篇各自解决什么

| 笔记 | 一句话贡献 | 没有解决的问题 |
|---|---|---|
| 01 NCSN | 多噪声尺度让 score 可学、Langevin 可混合 | 采样仍慢且依赖离散 schedule |
| 02 DDPM | 闭式加噪 + $\varepsilon$-MSE 得到稳定高质量生成 | 1000 步反向链成本高 |
| 03 Flow Matching | simulation-free 地直接回归 velocity，并开放路径选择 | 生成仍需 ODE solver，路径/耦合影响曲率 |
| 04 CFG | 无外部分类器地控制 fidelity-diversity | 每步双前向，强 guidance 损失覆盖度 |
| 05 Song SDE | 统一 NCSN/DDPM，并给出 reverse SDE 与 PF-ODE | solver 和 likelihood 仍昂贵、超参多 |
| 06 Drifting | 用训练时间 field 原生学习一步生成器 | 训练依赖 kernel、强 encoder 和大批量，理论仍受限 |

## 11 · 落到 Drifting proposal 的挂钩

- **Identifiability**：Drifting 的 $V\equiv 0 \Rightarrow q=p$ 目前只有 heuristic；类比 kernel MMD 的 characteristic kernel 假设可给出严格证明的候选路径。
- **训练动态的连续极限**：把 SGD 迭代抬升为 $\partial_\tau q + \nabla\cdot(q V) = 0$，可能揭示 Drifting 是某泛函的 $W_2$ gradient flow；与 Song 2021 的 PF-ODE 形成对偶（物理时间 vs 训练时间）。
- **Few-step 拓展**：把 1-NFE Drifting 拓展成 $K$ 层级 $V^{\left(1\right)},\dots,V^{\left(K\right)}$，介于 Flow Matching 与 Drifting 之间的连续插值。
- **Path-free vs path-based 的形式关系**：能否把 PF-ODE 从 $t=T$ 到 $t=0$ 的积分算子写成 $f_\theta$，从而把 Drifting 表达为 Flow Matching 的"整段积分极限"？
