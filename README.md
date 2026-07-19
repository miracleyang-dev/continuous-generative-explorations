# continuous-generative-explorations

Reading notes and small experiments on continuous-time generative models —
diffusion, score-based SDE, flow matching, rectified flow.

Companion repo to
[`arc-agi-explorations`](https://github.com/miracleyang-dev/arc-agi-explorations).
Same style: one folder per phase, one note per paper, code lives under
`scripts/` or `models/` when it appears.

Started during my summer research internship at HKUST (Jul – Aug 2026),
after redirecting focus from ARC-AGI to continuous-time generative modelling
at my advisor's suggestion. The immediate goal is to build a working
understanding of the **diffusion ⇄ flow matching** unification (via the
drift / velocity view) before touching any new method.

## Layout

```
notes/       Paper-reading notes (one .md per paper)
scripts/     Utilities (added when first script lands)
models/      Minimal reproductions (added when first repro lands)
```

## Planned notes (in reading order)

- `notes/01_song_sde.md`  — Song et al. 2021, *Score-Based Generative
  Modeling through Stochastic Differential Equations* (ICLR 2021).
  Foundational: unifies diffusion / score matching under a single SDE
  framework; introduces the probability-flow ODE that bridges to flow
  matching.
- `notes/02_generative_modeling_via_drifting.md`  — the paper my advisor
  handed me. Read after Song 2021 so the drift-term language lines up.

## Setup

To be added once the first script or model reproduction lands (likely
mirror the `.venv` + `requirements.txt` pattern from
`arc-agi-explorations`).

## References

- Song, Y. et al. (2021). *Score-Based Generative Modeling through
  Stochastic Differential Equations*. ICLR 2021.
  arXiv:[2011.13456](https://arxiv.org/abs/2011.13456)
- Ho, J. et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
  arXiv:[2006.11239](https://arxiv.org/abs/2006.11239)
- Lipman, Y. et al. (2022). *Flow Matching for Generative Modeling*. ICLR 2023.
  arXiv:[2210.02747](https://arxiv.org/abs/2210.02747)
- Liu, X. et al. (2022). *Rectified Flow*. ICLR 2023.
  arXiv:[2209.03003](https://arxiv.org/abs/2209.03003)

---

Maintained by [@miracleyang-dev](https://github.com/miracleyang-dev).
Work in progress; notes may be rewritten without notice.
