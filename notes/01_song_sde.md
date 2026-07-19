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

（待读完填）

## 2 · 关键公式

- Forward SDE:  $dx = f(x,t)\,dt + g(t)\,dW$
- Reverse SDE:  $dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar W$
- Probability flow ODE:  $dx = [f(x,t) - \tfrac12 g(t)^2 \nabla_x \log p_t(x)]\,dt$

（每条读到时补上：符号含义、推导来源、直觉）

## 3 · 主要贡献

（读完填 3-5 条）

## 4 · 我不理解的点

（读的过程中随时记）

## 5 · 跟 diffusion / flow matching 的连接

（读完 §4 后填 —— 尤其是 probability flow ODE 那节）

## 6 · 复现想法

（如果打算写代码，记在这里；先不做）

---

**状态**：未开始
**开始日期**：
**完成日期**：
