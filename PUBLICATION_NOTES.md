# Publication notes: novelty landscape + multimode-herald model spec

## 1. Novelty / literature — where your experiment sits

The theory of photon-subtracted (and added) squeezed states in SU(1,1) interferometers is
**well developed**; the matching **experiment is not**. That gap is your opening.

| work | source | interferometer | subtraction? | read-out | type |
|---|---|---|---|---|---|
| Machado, Thekkadath *et al.* [arXiv:2606.30761] | TMSV (Type-II) | **SU(1,1)** nonlinear | no | **PNR/TES joint** | **experiment** |
| Thekkadath *et al.*, npj QI 6,89 (2020) [arXiv:2006.08449] | heralded Fock | linear (MZ) | (Fock prep, not subtraction) | PNR/TES | experiment |
| photon-subtracted TMSV in SU(1,1): JOSA B 29,2581 (2012); [arXiv:2311.14612]; Optik (2023) | TMSV | SU(1,1) | **yes** | homodyne/parity | **theory** |
| "phase-sensitive non-Gaussianity", PRA 105,042607 (2022) | sqz | SU(1,1)/MZ | **yes** | — | **theory** |
| photon subtraction in MZ [arXiv:2505.04499] (2025) | sqz | linear | yes | — | theory |
| multi-photon subtraction, telecom [arXiv:2301.09871]; ps cat states [arXiv:2606.24002] | sqz | none (state gen.) | yes | homodyne/TES | experiment |

**The specific combination in your data — experimental, PNR-resolved, phase-scanned
photon *subtraction between the squeezers of a nonlinear (SU(1,1)) interferometer* —
does not appear to have been measured.** The theory proposals (esp. PRA 105,042607 on
"phase-sensitive non-Gaussianity" and the SU(1,1)+subtraction QCRB papers) are exactly
what your experiment would be the **first realization of**. That is a legitimate, if
specialist-level, novelty hook — provided (a) the subtracted-state model is completed and
(b) the claim is framed as **state engineering + measured non-Gaussian statistics /
Fisher-information structure**, NOT as a sub-shot-noise sensing record (the loss budget
does not support the latter — see RESULTS.md §4).

**Defensible headline options (pick one, in decreasing safety):**
1. *"PNR-resolved joint statistics of heralded photon-subtracted states in a nonlinear
   interferometer"* — a measurement/characterization paper. Safe, novel-by-combination.
2. *"Tuning photon-number correlations from bunched to antibunched by heralded subtraction
   inside a nonlinear interferometer"* — leads with the g²: 4.0 → 0.5 result + CS
   violation. Crisp, physical.
3. *"First measurement of phase-sensitive non-Gaussianity in an SU(1,1) interferometer"* —
   strongest, but requires matching a specific theory proposal and the completed model.

## 2. Multimode-herald model — spec + feasibility (closes the main gap)

**Problem.** The single-D-mode subtraction `d^{N1}` fits N₁=0 (χ²≈37, η₃=0.50, η₄=0.58)
but fails N₁=1,2 with shared parameters (χ²~10³–10⁴), the free fits demanding a ~3×
lower efficiency. Cause: the herald photon lives in a **partly distinct spectral/spatial
mode**, so "N₁ clicks" does not implement `d^{N1}` on the one measured mode.

**Feasibility (already tested).** Fixing the detectors at the N₁=0 values and modelling the
heralded joint as *unsubtracted beam + a small subtracted admixture* drops χ² from
**12500 → 303** (N₁=1) — with η held at 0.5. So the multimode picture is the right physics
and the efficiency inconsistency disappears. The crude 2-component proxy says the herald
subtracts from only ~a few % of a highly multimode beam.

**Proper model.** The PDC populates `K` Schmidt modes `{φ_j}` with weights `{λ_j}`
(Σλ_j²=1). Write the herald-detector mode as an overlap vector `{c_j}` on these modes.
Each Schmidt mode independently runs the seeded circuit (D single-mode squeeze with gain
`λ_j r₁` → phase → H-V two-mode squeeze with gain `λ_j r₂`). Detected photons at D3/D4
are the **sum over modes**, so the joint distribution is a **convolution** over Schmidt
modes (independent modes add in photon number):

```
 P(n3,n4 | N1) =  ⊛_j  P_j(·)         (2-D convolution over j)
   where the HERALDED mode(s) carry the subtraction weighted by the herald overlap |c_j|²,
   and the SPECTATOR modes carry the unsubtracted seeded circuit; all share eta3, eta4.
```

A tractable two-effective-mode reduction (recommended first implementation):

* **mode H (heralded):** overlap fraction `w = Σ_j |c_j|²`-weighted; runs the seeded
  circuit *with* `subtract N1`; effective gains `r1_H, r2_H`.
* **mode S (spectator):** the remaining `1-w`; runs the seeded circuit *without*
  subtraction; effective gains `r1_S, r2_S` fixed by the N₁=0 fit and the Schmidt weights.
* `P(n3,n4|N1) = P_H(subtract N1) ⊛ P_S(no subtract)`, shared `eta3, eta4` from N₁=0.

Free parameters beyond the (fixed) N₁=0 source/detectors: the herald overlap `w`
(equivalently the effective Schmidt number `K≈1/w`) and the phase offset — 2 per
condition. The convolution (not the classical mixture used in the feasibility test) should
push χ² toward the N₁=0 value.

**Independent cross-check available in your data:** the herald-arm `g⁽²⁾`. A single-mode
squeezed seed gives `g²=3+1/n̄`; `K` equal Schmidt modes give `g²≈2+1/K`… the measured
herald `g²=11.7` and the fitted `w` must be mutually consistent — that over-determines `K`
and validates the multimode picture without free knobs.

**Implementation sketch (fits in `ni_twomode.py`):**
```
def seeded_pn_multimode(phi, params, N1, w, D):
    P_H = seeded_pn(phi, {**params, 'r1':r1_H,'r2':r2_H}, N1=N1, D=D)   # heralded mode
    P_S = seeded_pn(phi, {**params, 'r1':r1_S,'r2':r2_S}, N1=0,  D=D)   # spectator
    return conv2d(P_H, P_S)                                             # per phase, then truncate
```
with `eta3, eta4` applied once to the convolved output. Fit `w` (shared across N₁=1,2 —
it is a property of the herald optics, not of N₁) + per-condition `phi0`. If N₁=1 and N₁=2
fit with a *single shared* `w` and the N₁=0 detectors, the subtracted-state model is
complete and the paper's central objects are fully characterized.

## 3. Minimal path to a submittable paper

1. Implement the multimode-herald convolution model above; fit N₁=0,1,2 (both read-outs)
   with shared `w`, `eta3`, `eta4`. Target: all χ²ᵥ ~ O(1–10) with consistent parameters.
2. Add single-pass gain/efficiency calibration (as in the reference, App. F) to remove
   residual degeneracy.
3. Report: (i) validated two-mode source + herald model; (ii) CS violation R(φ) and the
   g² tuning 4.0→0.5 vs N₁; (iii) CFI from the true joint vs total vs click, and the
   Fisher redistribution across Fock sectors under subtraction.
4. Frame per headline option 1 or 2; do not claim sub-SNL sensing.
