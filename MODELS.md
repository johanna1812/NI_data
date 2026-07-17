# Read-out models: PBS separation vs 50:50 (one two-mode state, one beam-splitter knob)

Your setup is **one two-mode state** (two polarisation modes `a`, `b` through the
nonlinear interferometer) read out through a **variable beam splitter** set by the HWP
before the PBS. There are not two different physical models — there is one model with a
beam-splitter angle:

| data set | HWP | model setting | what the detectors see |
|---|---|---|---|
| **PBS separation** | aligned | `theta_bs = 0` | the two polarisation modes `a`, `b` directly |
| **50:50** | 22.5° | `theta_bs = pi/4` | `(a±b)/√2` (the modes mixed 50:50) |

Both are `ni_twomode.two_mode_pn(phi, params, N1, ...)` with the **same physical
parameters**, only `theta_bs` differs. So the right way to use them is to **fit both
data sets jointly** with shared squeezing/gain and per-set detector efficiencies — the
PBS set pins the two-mode correlations, the 50:50 set pins the marginals.

```python
import ni_twomode as T, math
p = T.preset("PBS")            # theta_bs = 0
p = T.preset("5050")           # theta_bs = pi/4
P = T.two_mode_pn(phi_array, p, N1=0)     # -> P(n3,n4) per phase
T.moments(P.sum(0))            # g2_D3, g2_D4, g11, R=CSI ratio
```

## What the current (repo) data tells us about the source

The repo data is the **PBS** set. Its D3-D4 cross-correlation violates Cauchy-Schwarz
(`R = g11²/(g2₃g2₄) = 18`, see `fig9`). Two rigorous consequences:

1. **The squeezing is genuinely two-mode (parametric, pair-creating `a†b†`).** A
   single-mode squeezer — even one whose squeezed polarisation is rotated to an angle α
   relative to the PBS axes (`sq="rotated"`) — can **never** give `R > 1`: a split single
   mode cannot have cross-correlation exceeding its own auto-correlation. Only `a†b†`
   pair creation does. (Checked over α ∈ [0,45°]: R ≤ 0.9 always.)

2. **There is also a single-mode squeezing component.** Pure two-mode squeezing gives each
   arm thermal statistics `g² = 2`, but the data shows `g² ≈ 4-4.5`. Raising `g²` above 2
   requires a single-mode `a†²+b†²` term. So the source is a **general two-mode Gaussian
   squeezer** = two-mode gain (`rt`, sets `R`) **plus** single-mode gain (`rs`, raises
   `g²`). This is `sq="general"` in the model, with generator
   `rt(ab−a†b†) + ½rs(a²−a†² + b²−b†²)`.

**The general two-mode model fits the PBS data well** (`fit_general`, N₁=0):

| model | χ²ᵥ | (g²₃, g²₄, g⁽¹,¹⁾, R) | η₃, η₄ |
|---|---|---|---|
| single-mode passive split | ~200 | fails | 0.06 (absurd) |
| pure two-mode (`a†b†`) | ~107 | g²=2 (too low) | — |
| **general (`a†b†` + `a†²`)** | **~22** | **(3.7, 4.1, 16.6, 18) ≈ data (4.0, 4.5, 18, 18)** | **0.47, 0.54** |

Fitted gains: two-mode `rt≈0.13`, single-mode `rs≈0.04`, second stage `r2≈0.06`.
The recovered detection efficiencies **η₃=0.47, η₄=0.54 match the reference paper's TES
efficiencies (0.54, 0.58)** — a strong independent validation. (The single-mode-split
model was forced to ~6% because it had to hide the pair correlations inside "loss".)
The coarse tension mentioned above (raising `rs` lowers `R`) is resolved once the
detection efficiency and phase are free: the fit reaches `g²≈4` **and** `R≈18` together.

## The four building-block behaviours (model, N1=0, PBS vs 50:50)

| squeezer | PBS `R` | 50:50 `R` | note |
|---|---|---|---|
| `single` (independent a,b) | 0.02 | 0.38 | never violates CS |
| `rotated` (angle α) | ≤0.9 | — | never violates CS |
| `twomode` (`a†b†`) | 15.7 | 0.02 | violates CS in PBS only; `g²=2` |
| `general` (`a†b†`+`a†²`) | tunable | tunable | needed to match `g²≈4` **and** `R` |

So: your **PBS** data ⇒ `sq="general"`, `theta_bs=0`. Your **50:50** data ⇒ same
`sq="general"`, `theta_bs=pi/4` (there the pair correlations rotate into the marginals and
the earlier single-mode marginal model of `ni_analysis.py` is the effective description).

## Functions

* `ni_twomode.two_mode_pn(phi, params, N1, nmax, D)` — detected `P(n3,n4)` vs phase, for any
  squeezer type / `theta_bs` / heralded subtraction `N1` / per-detector loss+background.
* `ni_twomode.preset("PBS"|"5050", **overrides)` — parameter dict for either read-out.
* `ni_twomode.moments(P)` — `g²(D3), g²(D4), g⁽¹,¹⁾, R`.
* `ni_twomode.cs_ratio(params, N1)` — the CS ratio a configuration predicts.
* `ni_twomode.fit_twomode(pooled_ns, N1, sq, theta_bs, D)` — fit one condition's binned,
  drift-compensated joint matrices (use `ni_joint.pool_joint` to build `pooled_ns`).

## Grid results (`analysis_twomode_grid.py`, PBS/UP)

* **N₁=0 (no subtraction):** the general two-mode model fits excellently — χ²ᵥ≈22,
  η₃=0.47, η₄=0.54 (≈ reference TES), rt≈0.13, rs≈0.04. The data/model joint heatmaps
  agree (`fig10`).
* **CFI from the TRUE two-mode joint** (N₁=0): F_joint=5.1×10⁻³ **> F_total=4.8×10⁻³
  (+5%) > F_1det=2.7×10⁻³ (×2)**. So the pair correlations *do* add phase information
  over the total photon number — exactly the effect a passive split cannot give
  (there F_joint=F_total to the digit). This is the metrological payoff of the two-mode
  read-out.
* **N₁=1,2 (subtracted):** fixing the shared detectors + source and applying `a^{N1}`
  to one output mode **fails** (χ²~10⁴). Diagnosis: in this setup the **herald (Det 1) is
  a separate mode**, so heralding on N₁ is a *projective conditioning of a 3-mode state*
  (herald ⊗ signal ⊗ idler), not `a^{N1}` on an output mode. The subtracted-state model
  therefore needs the herald mode added explicitly — this is the next modelling step
  (see below). The single-mode marginal analysis of the subtracted data (figs 1–8)
  remains valid as a *marginal* description.

## Recommended next step

1. **Add the herald mode** (3-mode: herald ⊗ a ⊗ b) so the subtracted conditions
   (N₁≥1) are modelled by projecting on the herald photon number, not by `a^{N1}`. This
   is what makes the N₁=1,2 two-mode fits work.
2. Fit the **PBS and 50:50 data sets jointly**, sharing `(rt, rs, r2, phase)` and using
   separate `(eta3, eta4)` per set, with `theta_bs` fixed to 0 and pi/4 respectively. That
   over-determines the two-mode state and cleanly separates the two-mode gain (from the PBS
   correlations) from the single-mode gain (from the 50:50 marginals) — and then the CFI
   is evaluated from the true joint statistics, where the correlations add phase
   information over the total (already shown for N₁=0: +5%).
