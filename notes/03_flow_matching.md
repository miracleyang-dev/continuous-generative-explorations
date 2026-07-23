# 03 · Lipman et al. 2023 — Flow Matching for Generative Modeling

- 出处：ICLR 2023
- arXiv：<https://arxiv.org/abs/2210.02747>
- 日期：7.16

## 阅读结论

- **核心问题**：CNF 表达力强，但最大似然训练要在训练环中求解 ODE 并计算散度，成本高。
- **核心方法**：预先指定一条可采样的条件概率路径，直接用 MSE 回归其条件 velocity；边缘化后得到正确的边缘 velocity。
- **关键自由度**：训练目标与路径选择解耦，既可复现 diffusion path，也可采用更直的 OT conditional path。
- **历史位置**：Flow Matching 把“学 score 再构造 ODE”改成“直接学 ODE velocity”，是 diffusion 与 rectified flow 之间的主要接口。

## 1. 出发点：Continuous Normalizing Flow（CNF）

用一条 ODE 定义可逆映射：

$$
\frac{d\phi_t(x)}{dt} = v_t(\phi_t(x)),\qquad \phi_0(x) = x
$$

$v_t: \mathbb{R}^d \to \mathbb{R}^d$ 是时变 velocity field。由 $\phi_t$ 推前一个先验 $p_0$ 得到路径分布 $p_t = [\phi_t]_\# p_0$。

对应连续性方程（continuity / transport equation）：

$$
\partial_t p_t(x) + \nabla\cdot(p_t(x)\,v_t(x)) = 0
$$

**传统 CNF 训练**：最大化 $\log p_1(x)$，要跑一遍 ODE + 算 divergence（Neural ODE, Chen et al. 2018）。开销大，训练不稳定。

## 2. Flow Matching 的目标

**如果**已知一个能把 $p_0$ 传输到 $p_1 = p_{\text{data}}$ 的目标 velocity field $u_t(x)$，则回归它：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1]}\,\mathbb{E}_{x\sim p_t(x)}\bigl[\|v_\theta(x, t) - u_t(x)\|^2\bigr]
$$

问题：$u_t(x)$ 和 $p_t(x)$ 都不知道。作者用"条件化 + marginalization"绕过。

## 3. Conditional Flow Matching（CFM）

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
\bigl[\|v_\theta(x, t) - u_t(x\mid z)\|^2\bigr]
$$

**关键定理**：$\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$，两个 loss 对 $\theta$ 的梯度相同。证明只用 MSE 的展开 + 边缘化：

$$
\mathcal{L}_{\text{FM}} - \mathcal{L}_{\text{CFM}}
= \mathbb{E}_{t, x}\bigl[\|u_t(x)\|^2 - \mathbb{E}_z\|u_t(x\mid z)\|^2\bigr]
$$

右端不含 $\theta$。**所以只需要能采样 $p_t(x\mid z)$ 并计算 $u_t(x\mid z)$**，就能训。

## 4. Gaussian 条件路径

取

$$
p_t(x\mid x_1) = \mathcal{N}\bigl(x;\,\mu_t(x_1),\,\sigma_t(x_1)^2 I\bigr)
$$

任意光滑的 $\mu_t, \sigma_t$，满足 $p_0 = \mathcal{N}(0,I),\ p_1 \approx \delta_{x_1}$。

**Thm 3**：对论文选定的仿射 flow map，定义该 map 的唯一条件 velocity 为：

$$
u_t(x\mid x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}\,(x - \mu_t(x_1)) + \mu_t'(x_1)
$$

其中 $'$ 表示对 $t$ 求导。

### 4.1 与 diffusion 路径的对应

选 $\mu_t(x_1) = \alpha_t x_1,\ \sigma_t(x_1) = \sqrt{1 - \alpha_t^2}$（VP 型），可让 $p_t$ 与 VP-SDE 的边缘一致；此时 $u_t(x\mid x_1)$ 就是 probability flow ODE 的 velocity。

### 4.2 Optimal Transport conditional path

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
- 这里的“OT”指每个条件高斯到目标点的 Wasserstein-2 displacement interpolation；独立采样的 $(x_0,x_1)$ 并不等于求解数据边缘之间的全局最优耦合。

## 5. 训练、采样与似然

### 5.1 训练

```text
repeat:
    t ~ Uniform[0, 1]
    x_0 ~ N(0, I)
    x_1 ~ p_data
    x_t = (1 - t) * x_0 + t * x_1
    target = x_1 - x_0
    take gradient step on ||v_theta(x_t, t) - target||^2
```

训练过程中不求解 ODE，也不计算 divergence，因此是 simulation-free training。

### 5.2 采样

采完 $x_0 \sim \mathcal{N}(0, I)$，跑 ODE：

$$
\frac{dx}{dt} = v_\theta(x, t),\quad t: 0 \to 1
$$

- 用 RK4、Heun、Dopri5 等 solver。
- 采样成本取决于轨迹曲率、solver 和误差容限。原论文在统一 Dopri5 设置下报告约 122–193 NFE，仍少于其 DDPM 基线的约 262–274 NFE；不能把后续模型常见的 5–20 步直接归到这篇原论文。

### 5.3 Log-likelihood

CNF 仍可沿生成 ODE 使用瞬时变量替换公式计算 likelihood：

$$
\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \nabla\cdot v_\theta(x_t,t)\,dt
$$

Flow Matching 免掉的是**训练时**的 ODE 模拟与散度，不是 likelihood 评估时的积分。

## 6. 与 diffusion 的对比

| 维度 | Diffusion (DDPM / Song SDE) | Flow Matching |
|---|---|---|
| 训练目标 | 学 score / $\varepsilon$ | 学 velocity $v_\theta$ |
| 概率路径 | forward SDE 决定的 Gaussian 路径（VP/VE） | 任意可微 $p_t(x\mid z)$，可选 OT 直线 |
| 采样 | reverse SDE 或 PF-ODE | ODE only |
| 原论文统一评估 NFE | DDPM 约 262–274 | FM 约 122–193 |
| 训练稳定性 | 良好 | 良好（无 CNF 的 log-det 项） |

关系：Song 2021 的 PF-ODE 是 diffusion 路径下的 flow matching 特例（$v = f - \tfrac12 g^2 s_\theta$）。Flow matching 把路径**参数化**从 diffusion 中解耦，允许 OT / rectified 等更短路径。

## 7. 实验结果与局限

- CIFAR-10（unconditional）：FID **6.35**（OT-CFM），优于同结构 diffusion。
- ImageNet 32×32：OT-CFM FID **5.02**、122 NFE；对应 DDPM 为 FID 6.99、262 NFE。
- ImageNet 64×64：OT-CFM FID **14.45**、138 NFE；对应 DDPM 为 FID 17.36、264 NFE。

局限：

- CFM 的监督方差取决于条件路径与端点耦合；独立配对虽然简单，却不保证边缘流最直。
- 生成仍需数值积分，NFE 不是固定值；更严格的误差容限会增加计算。
- ODE 为确定性采样器，但确定性不自动意味着低成本或数值稳定，仍需检查 stiffness 与 solver 误差。

## 8. 核心符号速查

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
