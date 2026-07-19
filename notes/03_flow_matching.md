# 03 · Lipman et al. 2023 — Flow Matching for Generative Modeling

- 出处：ICLR 2023
- arXiv：<https://arxiv.org/abs/2210.02747>
- 日期：7.16

## 出发点：Continuous Normalizing Flow (CNF)

用一条 ODE 定义可逆映射：

$$
\frac{d\phi_t(x)}{dt} = v_t(\phi_t(x)),\qquad \phi_0(x) = x
$$

$v_t: \mathbb{R}^d \to \mathbb{R}^d$ 是时变 velocity field。由 $\phi_t$ 推前一个先验 $p_0$ 得到路径分布 $p_t = [\phi_t]_\# p_0$。

对应连续性方程（continuity / transport equation）：

$$
\partial_t p_t(x) + \nabla\cdot(p_t(x)\,v_t(x)) = 0 \tag{CE}
$$

**传统 CNF 训练**：最大化 $\log p_1(x)$，要跑一遍 ODE + 算 divergence（Neural ODE, Chen et al. 2018）。开销大，训练不稳定。

## Flow Matching 的想法

**如果**已知一个能把 $p_0$ 传输到 $p_1 = p_{\text{data}}$ 的目标 velocity field $u_t(x)$，则回归它：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1]}\,\mathbb{E}_{x\sim p_t(x)}\bigl[\|v_\theta(x, t) - u_t(x)\|^2\bigr] \tag{FM}
$$

问题：$u_t(x)$ 和 $p_t(x)$ 都不知道。作者用"条件化 + marginalization"绕过。

## 条件化：Conditional Flow Matching (CFM)

选一个条件变量 $z$（最常见 $z = x_1$，即目标数据点），定义

- 条件概率路径 $p_t(x\mid z)$：一族在 $t\in[0,1]$ 上从简单分布到点质量 $\delta_{x_1}$ 的插值；
- 条件 velocity field $u_t(x\mid z)$：驱动 $p_t(\cdot\mid z)$ 的场。

**边缘化关系**（Thm 1）：

$$
p_t(x) = \int p_t(x\mid z)\,q(z)\,dz
$$

$$
u_t(x) = \int u_t(x\mid z)\,\frac{p_t(x\mid z)\,q(z)}{p_t(x)}\,dz
= \mathbb{E}_{z\sim q(z\mid x, t)}[u_t(x\mid z)]
$$

即边缘 velocity 是条件 velocity 在后验 $q(z\mid x, t)$ 下的期望。

**CFM 目标**（Thm 2）：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\,z\sim q(z),\,x\sim p_t(x\mid z)}
\bigl[\|v_\theta(x, t) - u_t(x\mid z)\|^2\bigr] \tag{CFM}
$$

**关键定理**：$\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$，两个 loss 对 $\theta$ 的梯度相同。证明只用 MSE 的展开 + 边缘化：

$$
\mathcal{L}_{\text{FM}} - \mathcal{L}_{\text{CFM}}
= \mathbb{E}_{t, x}\bigl[\|u_t(x)\|^2 - \mathbb{E}_z\|u_t(x\mid z)\|^2\bigr]
$$

右端不含 $\theta$。**所以只需要能采样 $p_t(x\mid z)$ 并计算 $u_t(x\mid z)$**，就能训。

## Gaussian 条件路径

取

$$
p_t(x\mid x_1) = \mathcal{N}\bigl(x;\,\mu_t(x_1),\,\sigma_t(x_1)^2 I\bigr)
$$

任意光滑的 $\mu_t, \sigma_t$，满足 $p_0 = \mathcal{N}(0,I),\ p_1 \approx \delta_{x_1}$。

**Thm 3**：给出 $p_t(\cdot\mid x_1)$ 的**唯一**闭式条件 velocity（在 push-forward 意义下）：

$$
u_t(x\mid x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}\,(x - \mu_t(x_1)) + \mu_t'(x_1)
$$

其中 $'$ 表示对 $t$ 求导。

### 与 diffusion 路径的对应

选 $\mu_t(x_1) = \alpha_t x_1,\ \sigma_t(x_1) = \sqrt{1 - \alpha_t^2}$（VP 型），可让 $p_t$ 与 VP-SDE 的边缘一致；此时 $u_t(x\mid x_1)$ 就是 probability flow ODE 的 velocity。

## Optimal Transport 路径（本文主推）

取**直线插值**：

$$
\mu_t(x_1) = t\,x_1,\qquad \sigma_t(x_1) = 1 - (1 - \sigma_{\min})\,t
$$

$x_0 \sim \mathcal{N}(0, I)$，$x_1 \sim p_{\text{data}}$，令

$$
x_t = (1 - (1 - \sigma_{\min})\,t)\,x_0 + t\,x_1
$$

对应条件 velocity（对 $t$ 求导）：

$$
u_t(x_t \mid x_1) = x_1 - (1 - \sigma_{\min})\,x_0
$$

$\sigma_{\min} \to 0$ 时：

$$
u_t(x_t\mid x_1) = x_1 - x_0
$$

**极简训练目标**：

$$
\mathcal{L}_{\text{OT-CFM}}(\theta) = \mathbb{E}_{t\sim U[0,1],\,x_0\sim\mathcal{N}(0,I),\,x_1\sim p_{\text{data}}}
\bigl[\|v_\theta(t x_1 + (1-t) x_0,\,t) - (x_1 - x_0)\|^2\bigr]
$$

- 路径是直线，velocity 是常向量 $x_1 - x_0$。
- 与 Rectified Flow (Liu et al. 2022) 一致（独立工作）。

## 采样

采完 $x_0 \sim \mathcal{N}(0, I)$，跑 ODE：

$$
\frac{dx}{dt} = v_\theta(x, t),\quad t: 0 \to 1
$$

- 用 RK4、Heun、Dopri5 等 solver。
- 步数远少于 diffusion 的 1000 步；OT 路径下少至 5–20 步可保持质量。

## 与 diffusion 的对比

| 维度 | Diffusion (DDPM / Song SDE) | Flow Matching |
|---|---|---|
| 训练目标 | 学 score / $\varepsilon$ | 学 velocity $v_\theta$ |
| 概率路径 | forward SDE 决定的 Gaussian 路径（VP/VE） | 任意可微 $p_t(x\mid z)$，可选 OT 直线 |
| 采样 | reverse SDE 或 PF-ODE | ODE only |
| 步数 | 通常 50–1000 | 通常 5–100 |
| 训练稳定性 | 良好 | 良好（无 CNF 的 log-det 项） |

关系：Song 2021 的 PF-ODE 是 diffusion 路径下的 flow matching 特例（$v = f - \tfrac12 g^2 s_\theta$）。Flow matching 把路径**参数化**从 diffusion 中解耦，允许 OT / rectified 等更短路径。

## 实验结果（要点）

- CIFAR-10（unconditional）：FID **6.35**（OT-CFM），优于同结构 diffusion。
- ImageNet 32/64：OT-CFM 在少步采样（10–20 NFE）下明显优于 DDPM。
- 采样步数-质量曲线：FM 曲线更平，diffusion 在低 NFE 掉得快。

## 核心符号速查

| 符号 | 含义 |
|---|---|
| $\phi_t$ | ODE 定义的 flow |
| $v_t(x)$ | 边缘 velocity field |
| $u_t(x\mid z)$ | 条件 velocity field |
| $p_t(x)$ | 边缘概率路径 |
| $p_t(x\mid z)$ | 条件概率路径 |
| $q(z)$ | 条件变量分布（$q(x_1) = p_{\text{data}}$） |
| $x_0, x_1$ | 端点样本，$x_0\sim\mathcal{N}(0,I),\ x_1\sim p_{\text{data}}$ |
| $\mu_t, \sigma_t$ | 条件高斯路径参数 |

---

状态：主文完
