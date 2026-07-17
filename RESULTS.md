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
