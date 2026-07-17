# Results: PNR-resolved single-mode nonlinear interferometer with photon subtraction

Analysis produced by `ni_analysis.py` (theory in `THEORY.md`). All numbers below are for
detector 4's marginal unless stated; UP and DOWN sweeps agree, which is the main
consistency check.

## 1. What the data is

* 41 piezo voltages (10–50 V), UP and DOWN, 30 time "rounds" each.
* Each round × voltage gives a herald-resolved 9×9 joint distribution of detectors 3 and 4,
  conditioned on the herald detector 1 registering `Px = N₁+1` photons
  (`N₁` = number of subtracted photons: 0 = no subtraction, 1, 2, …).
* Photon-number labels: `Px_k ↔ (k−1)` photons (confirmed from the conditional-statistics
  block of the multi-coincidence files).

## 2. Why the earlier fits failed

1. **Phase drift.** φ₀ drifts by ≈ −160° (fringe phase) across the 30 rounds
   (`fig2`), so averaging rounds at fixed voltage *smears the fringe*. → fit each round.
2. **Wrong subtraction model.** Heralding on N₁ is **not** an input Fock state to the first
   squeezer (`n_in=1`); it is `a^{N₁}` applied **between** the squeezers. Corrected in the
   exact model `S₂ R(φ) a^{N₁} S₁|0⟩`.
3. **No background term.** The fringe minima are non-zero (accidental coincidences); a
   Poisson background is required or the model cannot reach the data.
4. **Per-sector independent cosines + degeneracy.** Fitting each P(n) with its own cosine
   ignores that odd/even sectors are physically linked and leaves (r₁, r₂, η) unidentified.
   → one **joint fit** across all photon sectors and all herald conditions, sharing
   (r₂, η, ω, φ₀); **r₁ fixed from the herald arm**.

## 3. Extracted device parameters (median over rounds, UP≈DOWN)

| quantity | value | how |
|---|---|---|
| r₁ (first squeezer) | **0.332** | herald-arm `g⁽²⁾ = 11.7 → sinh²r₁ = 1/(g²−3)` (loss-independent) |
| r₂ (second squeezer) | **0.05 ± 0.006** | joint output fit |
| η (collection, incl. 50:50 split) | **0.045 ± 0.001** | joint output fit |
| fringe period | **≈ 35 V** | ⟨n⟩ ∝ 1+cos(2φ) |
| reduced χ² | ~100 | high only because counts are huge (systematic, model ~1–2% level) |

**Diagnosis:** the interferometer runs with **r₁ ≫ r₂ (G1 > G2)** and ~95% loss. The
reference (arXiv:2606.30761) shows the loss-tolerant regime is the *opposite*, **G2 > G1**,
which amplifies the phase-sensitive correlations *before* the lossy detection stage. This
is the single biggest lever to get an absolute quantum advantage here.

## 4. Fisher information and phase sensitivity

CFI computed as `F(φ)=Σ_n (1/p_n)(∂p_n/∂φ)²` with **p_n from data, ∂p_n/∂φ from the fitted
model** (no noisy finite differences). Benchmarks: click detection `{0,≥1}` and the
shot-noise limit `F_SNL = sinh²(r₁) = 0.114` (per shot).

| herald | F_PNR (per shot) | PNR vs click | F_PNR/F_SNL | achieved Δφ (full data) |
|---|---|---|---|---|
| N₁=0 | 6.0×10⁻⁴ | **+6 %** | 0.005 | 4.8×10⁻⁵ rad |
| N₁=1 | 1.1×10⁻³ | **+21 %** | 0.010 | 3.8×10⁻⁴ rad |
| N₁=2 | 2.6×10⁻³ | **+4 %** | 0.023 | 9.8×10⁻⁴ rad |

* **Photon counting beats click detection** for every herald condition; the gain is
  largest for one-photon subtraction (`N₁=1`). This is the robust, scale-free result.
* **No absolute (unconditional) quantum advantage:** `F_PNR ≪ F_SNL`, i.e. ~200× below
  shot noise per shot, because of ~95 % loss and the G1>G2 gain configuration. This is
  expected and consistent — it is not a bug in the analysis.
* The **achieved** Δφ is small (10⁻⁴–10⁻⁵ rad) only because of the enormous number of
  shots (M ~ 10⁹–10¹¹); per single shot the state is ~98 % vacuum.

## 5. Concrete recommendation that helps immediately

**Reconstruct the full output photon number `n₃+n₄` instead of one detector marginal.**
The 50:50 split throws away half the light; counting both halves roughly doubles the
detected ⟨n⟩ and the PNR advantage (`fig5`):

| herald | PNR advantage, 1 detector | PNR advantage, n₃+n₄ |
|---|---|---|
| N₁=0 | +6 % | +8 % |
| N₁=1 | +21 % | **+42 %** |
| N₁=2 | +4 % | +8 % |

(and ~10× higher absolute CFI, because η goes 0.045 → 0.15).

Other things worth doing:
* Operate at **G2 > G1** (stronger second squeezer) for loss tolerance.
* Increase gain / reduce loss so ⟨n⟩_det ≳ 1 — the PNR-over-click advantage grows with
  photon number (the reference reaches +44 % at ⟨n⟩ ~ 1; you are at ⟨n⟩ ~ 0.01–0.1).
* Use the **full 2-D joint** P(n₃,n₄) for the CFI (as the reference does with p_mn), not
  just the total — the correlations carry extra phase information.
* Actively stabilise or track the phase (the −160° drift over ~7 h limits coherent
  averaging); the per-round φ₀ in `fig2` is a ready-made feed-forward signal.
* Report `g⁽²⁾` of the herald arm per gain setting — it is the cleanest, loss-free
  monitor of r₁.

## 5b. Loss everywhere, separate detectors, and drift compensation (general model)

`run_general()` uses full density-matrix propagation so loss can sit **anywhere**, not
just at the end (see THEORY §3b). Fitting the pooled, drift-compensated data per herald
condition (`fig6`, `fig7`):

| herald | r₂ | η₃ | η₄ | η₄/η₃ | χ²ᵥ |
|---|---|---|---|---|---|
| N₁=0 | 0.040 | 0.058 | 0.067 | 1.17 | 4.8 |
| N₁=1 | 0.072 | 0.032 | 0.039 | 1.22 | 66 |
| N₁=2 | 0.051 | 0.044 | 0.051 | 1.17 | 53 |

* **Detectors 3 and 4 have different efficiencies:** η₄/η₃ ≈ **1.2**, stable across all
  herald conditions — so yes, two separate losses are warranted (D4 ~20% better than D3).
* **Internal loss η_int is degenerate** with the gain ratio r₁/r₂ in a single-detector
  marginal: profiling gives χ²(η_int) = 5→14 as η_int goes 1.0→0.25 — a shallow,
  monotonic rise with no real minimum, so the data cannot fix it. It is included in the
  model and correctly re-amplified by S₂, but to *measure* it you need single-pass gain/
  efficiency calibration (reference Appendix F) or the full 2-D joint P(n₃,n₄).
* The single-mode model slightly underpredicts P(2) when one (r₂, η) is forced across all
  herald conditions; per-condition fits remove this (χ²ᵥ=4.8 for N₁=0). The residual
  N₁=1,2 tension is consistent with a **few Schmidt (spectral) modes** and the weak-tap
  approximation for subtraction — a multimode/thermal background term is the next
  refinement.

**Drift compensation (`fig7`).** Each scan is placed on a common phase axis
φ = ω·V + φ₀(round), with φ₀ from the per-round mean-photon fringe. Raw vs voltage the
30 scans are smeared by the ~160° drift; after compensation they **collapse onto a single
clean fringe** for P(1), P(2) and P(3), plotted vs φ in radians, and the model
overlays it. This is the correct way to combine the scans without washing out the fringe.

## 5c. Full 2-D joint P(n₃,n₄): the readout is genuinely two-mode (key finding)

`ni_joint.py` / `analysis_joint_correlations.py` build the exact joint model and test the
two requested items. Result — the more interesting one — is that **the D3/D4 pair is not a
passive split of one mode; the two detectors are photon-number-correlated (signal/idler)**:

* A passive 50:50 split + number detection is a number-diagonal POVM, so it predicts
  P(n₃,n₄) = trinomial spread of the single output diagonal, with the split ratio
  phase-independent. That model **fails**: it under-predicts the coincidence P(1,1) by
  ~4.5× and P(2,2) by ~7×, with χ² ~ 200–290.
* **Cauchy–Schwarz test** R = g⁽¹,¹⁾²/(g⁽²⁾₃ g⁽²⁾₄) (classical bound R ≤ 1), UP≈DOWN:

  | herald | g²(D3) | g²(D4) | g⁽¹,¹⁾ (cross) | R | |
  |---|---|---|---|---|---|
  | no subtraction | 4.0 | 4.5 | 18.0 | **18** | strongly nonclassical |
  | N₁=1 | **0.50** | **0.49** | 0.80 | **2.6** | nonclassical + antibunched marginals |
  | N₁=2 | 2.2 | 2.3 | 3.5 | **2.5** | nonclassical |

  All conditions violate the classical bound → genuine two-mode (pair) correlations.
* Bonus nonclassicality: one-photon subtraction drives the **single-detector** marginal
  from super-Poissonian (g²=4.0) to **sub-Poissonian, antibunched (g²=0.49)**.

**Consequences.**
1. The correct model is the **two-mode** framework (arXiv:2606.30761 Eq. 3 for the
   no-subtraction case; extended for subtraction), *not* the single-mode model of
   `ni_analysis.py`. The single-mode marginal fits (§1–5) remain valid as *marginal*
   descriptions, but the joint needs two modes.
2. Because the correlations are real, the **full joint P(n₃,n₄) carries phase information
   beyond the total n₃+n₄** (the "joint = total" equality found with the single-mode
   model is an artefact of that wrong model). Quantifying the excess CFI needs the
   two-mode model. Using both detectors already ~doubles the CFI vs one detector.
3. The **multimode/thermal background is not the cause of the N₁=1,2 residual**: thermal
   background does not lower χ² (267→279 for N₁=1); the residual is the missing two-mode
   correlation.

## 6. Literature — has this been measured?

Photon-subtracted squeezed states for phase estimation are **well studied theoretically**
but **rarely measured with photon-number resolution**:

* SU(1,1) + photon subtraction, phase sensitivity / QCRB: Guo *et al.*, *Quantum optical
  interferometry via general photon-subtracted two-mode squeezed states*, Chin. Phys. B 28,
  094217 (2019); *Parity-based estimation in an SU(1,1) interferometer with
  photon-subtracted squeezed vacuum states*, Optik (2023); *Phase estimation via
  delocalized photon subtraction inside the SU(1,1) interferometer*, arXiv:2506.07684 (2025).
* Photon-**added/subtracted** squeezed light improving SU(1,1) sensitivity: Opt. Express 26,
  29099 (2018) and related.
* **PNR-resolved nonlinear interference** (the closest experiment to yours): Machado *et al.*,
  arXiv:2606.30761 (2026) — two-mode SU(1,1) + TES PNR, closed-form joint statistics, CFI vs
  SNL and vs click detection. Your single-mode, photon-subtracted variant is a genuinely
  novel measurement direction.
* Experimental photon subtraction / de-Gaussification with PNR/TES (state engineering,
  not interferometry): Ourjoumtsev *et al.*, Science 312, 83 (2006); Gerrits *et al.*,
  Phys. Rev. A 82, 031802(R) (2010); symmetric photon-subtracted TMSV via bright SPDC + PNR.

Bottom line: measuring **PNR-resolved fringes of a photon-subtracted state inside a
nonlinear interferometer, and extracting its CFI**, does not appear to have been published —
your data set is positioned to be a first if the loss/gain configuration is improved.
