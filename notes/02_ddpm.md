# 02 · Ho, Jain, Abbeel 2020 — Denoising Diffusion Probabilistic Models (DDPM)

- 出处：NeurIPS 2020
- arXiv：<https://arxiv.org/abs/2006.11239>
- 日期：7.15

## 模型：离散马尔可夫链

### Forward process（固定，无学习参数）

给定 schedule $\{\beta_t\}_{t=1}^T,\ \beta_t \in (0,1)$：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\bigr)
$$

$$
q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t\mid x_{t-1})
$$

选 $\beta_t$ 足够小 + $T$ 足够大 → $q(x_T)\approx\mathcal{N}(0,I)$。

### 闭式的边缘与后验（关键引理）

令 $\alpha_t := 1 - \beta_t$，$\bar\alpha_t := \prod_{s=1}^t \alpha_s$。归纳可证：

$$
q(x_t\mid x_0) = \mathcal{N}\bigl(x_t;\,\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t) I\bigr) \tag{★}
$$

即 $x_t$ 可一步采样：$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ \varepsilon\sim\mathcal{N}(0,I)$。

后验（Bayes + 高斯代数）：

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\bigl(x_{t-1};\,\tilde\mu_t(x_t, x_0),\,\tilde\beta_t I\bigr)
$$

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,x_0
+ \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,x_t,\qquad
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t
$$

### Reverse process（学习）

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\bigl(x_{t-1};\,\mu_\theta(x_t, t),\,\Sigma_\theta(x_t, t)\bigr)
$$

DDPM 取 $\Sigma_\theta = \sigma_t^2 I$ 固定（$\sigma_t^2 = \beta_t$ 或 $\tilde\beta_t$），只学 $\mu_\theta$。

## 变分下界（VLB / ELBO）

$$
-\log p_\theta(x_0) \le \mathbb{E}_q\bigl[-\log p_\theta(x_0\mid x_1)\bigr] + \sum_{t=2}^T \mathbb{E}_q\bigl[D_{\text{KL}}(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t))\bigr] + D_{\text{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))
$$

记为 $L_0 + \sum_{t\ge 2} L_{t-1} + L_T$。

- $L_T$ 与 $\theta$ 无关（$q(x_T\mid x_0)$ 由 forward 决定，$p(x_T)$ 是先验高斯）。
- $L_0$ 是像素级 discrete decoder。
- $L_{t-1}$ 是两个高斯之间的 KL，闭式。

### 高斯之间的 KL 化简

两个高斯 $\mathcal{N}(\mu_1,\sigma^2 I),\ \mathcal{N}(\mu_2,\sigma^2 I)$ 的 KL：

$$
D_{\text{KL}} = \tfrac{1}{2\sigma^2}\|\mu_1 - \mu_2\|^2 + \text{const}
$$

于是

$$
L_{t-1} = \mathbb{E}_q\Bigl[\tfrac{1}{2\sigma_t^2}\,\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\Bigr] + C
$$

## $\epsilon$-parameterization（Ho 的关键改动）

由 (★) 反解：$x_0 = \tfrac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon)$。代入 $\tilde\mu_t$：

$$
\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\Bigr)
$$

参数化 $\mu_\theta$ 也用同一形式，只学 $\varepsilon_\theta(x_t, t)$：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\Bigr)
$$

代回 $L_{t-1}$：

$$
L_{t-1} = \mathbb{E}_{x_0, \varepsilon}\Bigl[
\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\,
\bigl\|\varepsilon - \varepsilon_\theta\bigl(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\,t\bigr)\bigr\|^2\Bigr]
$$

## $L_{\text{simple}}$（简化训练目标）

丢掉复杂权重系数，得到

$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t,\,x_0,\,\varepsilon}
\Bigl[\bigl\|\varepsilon - \varepsilon_\theta\bigl(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\,t\bigr)\bigr\|^2\Bigr]
$$

- 训练：随机采 $t\sim\text{Uniform}\{1,\dots,T\}$，采 $x_0$，采 $\varepsilon$，反传 MSE。
- 与加权 VLB 的差：$L_{\text{simple}}$ 相当于把每个 $t$ 的权重压平；作者报告经验上样本质量更好。

### 与 DSM 的等价

$$
\varepsilon = -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t\mid x_0)
$$

所以 $\varepsilon_\theta(x_t, t)$ 与 score $s_\theta(x_t, t)$ 差一个 $-1/\sqrt{1-\bar\alpha_t}$ 的尺度：

$$
s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}
$$

DDPM 的 $L_{\text{simple}}$ 即是加权 DSM，$\lambda(t) = 1 - \bar\alpha_t$（相对 $\|s-\nabla\log q\|^2$ 尺度）。

## 采样（Algorithm 2）

```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-bar_alpha_t) * eps_theta(x_t, t)) + sigma_t * z
return x_0
```

- $T = 1000$（原论文）。
- $\sigma_t^2 = \beta_t$ 或 $\tilde\beta_t$；作者报告两者接近。

## 训练（Algorithm 1）

```
repeat:
    x_0 ~ p_data
    t ~ Uniform{1, ..., T}
    eps ~ N(0, I)
    x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps
    take gradient step on || eps - eps_theta(x_t, t) ||^2
```

## 网络与超参

- 网络：U-Net + self-attention（16×16 分辨率处）+ sinusoidal timestep embedding。
- Schedule：linear $\beta_t$ from $10^{-4}$ to $0.02$，$T=1000$。
- 数据集：CIFAR-10、CelebA-HQ 256、LSUN。

## 实验结果（要点）

- CIFAR-10：FID **3.17**，Inception Score 9.46（unconditional）。
- 优于当时所有 likelihood-based 模型，接近 BigGAN。
- Bits/dim 2.99（用 $L_{\text{simple}}$ 训练时次优，用 VLB 训练时更紧）。

## 与相关工作的接口

- **NCSN (Song & Ermon 2019)**：不同参数化（$s_\theta$ vs $\varepsilon_\theta$）+ 不同 schedule（VE vs VP），但训练目标同族。
- **Song 2021（SDE）**：DDPM = VP-SDE 的 Euler–Maruyama 离散化，$\beta_t \approx \beta(t)\Delta t$。
- **Improved DDPM (Nichol & Dhariwal 2021)**：学 $\Sigma_\theta$、cosine schedule。
- **DDIM (Song, Meng, Ermon 2021)**：非马尔可夫 forward，一步跨多步；可用同一 $\varepsilon_\theta$。

## 核心符号速查

| 符号 | 含义 |
|---|---|
| $\beta_t$ | forward noise schedule |
| $\alpha_t = 1 - \beta_t$ | 单步保留系数 |
| $\bar\alpha_t = \prod_{s\le t}\alpha_s$ | 累计保留系数 |
| $q(x_t\mid x_0)$ | 边缘（闭式，(★)） |
| $\tilde\mu_t,\tilde\beta_t$ | 后验 $q(x_{t-1}\mid x_t, x_0)$ 参数 |
| $\varepsilon_\theta(x_t, t)$ | 噪声预测网络 |
| $\sigma_t^2$ | reverse 方差（固定，$\beta_t$ 或 $\tilde\beta_t$） |
| $L_{\text{simple}}$ | 无权重的 $\varepsilon$-MSE loss |

---

状态：主文完
