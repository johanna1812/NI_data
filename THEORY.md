# Analytical model of a single-mode nonlinear (SU(1,1)) interferometer
# with photon subtraction, PNR read-out

This note derives the photon-number distributions `P(n,φ)` that are measured in the
piezo (phase) scan, for two cases:

1. the bare nonlinear interferometer (two single-mode squeezers, no subtraction), and
2. the state obtained by **subtracting `N₁` photons between the two squeezers**
   (heralded on the herald detector, "Detector 1").

It also gives the classical Fisher information (CFI), the phase sensitivity, and the
shot-noise (classical) benchmark used in `ni_analysis.py`.

The layout follows the reference experiment of Machado *et al.*,
*"Improving the loss threshold for quantum advantage in photonic sensors by complete
photon counting"* (arXiv:2606.30761), but specialised to a **single spatial/polarisation
mode** (degenerate squeezers) instead of the two-mode signal/idler SU(1,1).

---

## 1. Set-up and conventions

The optical mode passes through

```
 |0⟩ ──[ S(r₁,θ₁) ]──[ subtract N₁ ]──[ R(φ) ]──[ S(r₂,θ₂) ]──[ loss η ]── PNR detect
        squeezer 1     herald tap      piezo       squeezer 2   collection
```

* Single-mode squeezing operator `S(ξ)=exp[½(ξ* a² − ξ a†²)]`, `ξ = r e^{iθ}`.
  In the Heisenberg picture `S†(ξ) a S(ξ) = a cosh r − a† e^{iθ} sinh r`.
* Phase (piezo) `R(φ)=exp(−iφ a†a)`, `R† a R = a e^{−iφ}`.
* `φ = ω·V + φ₀` maps piezo voltage `V` to optical phase (ω, φ₀ from the fit).
* Loss/collection modelled by a beam-splitter of transmission `η` (the 50:50 split onto
  detectors 3/4 and all downstream inefficiency are absorbed into `η`).
* A Poissonian background (dark counts / accidental coincidences) of mean `n_bg`
  is added at the detector, exactly as in the reference (their Appendix E).

---

## 2. Bare nonlinear interferometer (no subtraction)

### 2.1 Effective squeezing and the mean-photon fringe

Because `S₂ R(φ) S₁` is a Gaussian unitary acting on the vacuum with **no displacement**,
the output is again a (rotated) **squeezed vacuum**. Composing the Bogoliubov maps, the
output annihilation operator is `b = μ a + ν a†` with

```
μ(φ) = cosh r₁ cosh r₂ e^{−iφ} + sinh r₁ sinh r₂ e^{ i(θ₂−θ₁)} e^{ iφ}
ν(φ) = −( cosh r₂ sinh r₁ e^{iθ₁} e^{−iφ} + cosh r₁ sinh r₂ e^{iθ₂} e^{ iφ} )
```

The **mean output photon number** is `⟨n⟩_out(φ) = |ν(φ)|²`:

```
⟨n⟩_out(φ) = cosh²r₂ sinh²r₁ + cosh²r₁ sinh²r₂
             + ½ sinh(2r₁) sinh(2r₂) · cos(2φ + Δθ) ,     Δθ = θ₂ − θ₁.
```

This is a **pure sinusoid in the phase** that swings between

```
 ⟨n⟩_min = sinh²(r₁ − r₂)      (squeezers cancel: de-amplification)
 ⟨n⟩_max = sinh²(r₁ + r₂)      (squeezers add:   amplification)
```

Two facts to note:

* The fringe is periodic in `2φ` (a **photon pair** picks up twice the single-mode phase),
  so the period in `φ` is `π`. In voltage the period is `π/ω`.
* Linear loss multiplies the whole curve by `η` and adds `n_bg`, so the **detected**
  mean is `⟨n⟩_det(φ) = η·⟨n⟩_out(φ) + n_bg`. Loss does **not** change the fringe
  *visibility* `V = (⟨n⟩_max−⟨n⟩_min)/(⟨n⟩_max+⟨n⟩_min)`; a reduced visibility is caused by
  **gain imbalance** `r₁≠r₂` and by **background** — this is the first thing to fit.

### 2.2 Photon-number distribution

Define the effective squeezing `s(φ)` by `sinh²s(φ) = ⟨n⟩_out(φ)`. The **ideal**
(loss-free) distribution is squeezed-vacuum, i.e. even photon numbers only:

```
 P_id(2k, φ) = (2k)! / [2^{2k} (k!)²] · tanh^{2k}s(φ) / cosh s(φ),
 P_id(2k+1, φ) = 0,
 with  tanh²s(φ) = ⟨n⟩_out(φ) / (1 + ⟨n⟩_out(φ)).
```

With detection loss the state becomes a **zero-mean Gaussian (squeezed-thermal) state**
with `N=η⟨n⟩_out`, `|M|² = η²⟨n⟩_out(⟨n⟩_out+1)`, and the measured distribution is the
binomial-loss image of `P_id`,

```
 P(n, φ) = Σ_{m≥n}  C(m,n) η^n (1−η)^{m−n}  P_id(m, φ).       (★)
```

Equation (★) is what `ni_analysis.py` evaluates exactly (it is the single-mode analogue
of Eq. 3 of the reference, whose closed hypergeometric form comes from the same
characteristic-function integral). Loss fills in the **odd** sectors, which is why
`P(1,φ)` is non-zero in the data.

---

## 3. Photon subtraction between the squeezers (herald on N₁)

A weak tap between the squeezers sends a small reflection to the herald detector.
Detecting `N₁` photons there applies (in the weak-tap limit) the annihilation operator
`a^{N₁}` to the transmitted beam. The heralded, **unnormalised** state at the output is

```
 |ψ_{N₁}(φ)⟩ = S(r₂) · R(φ) · a^{N₁} · S(r₁) |0⟩.
```

(The phase between tap and second squeezer only multiplies the herald amplitude by a
global `e^{−iN₁φ}` and does not affect probabilities.) The ideal distribution is

```
 P_id(n, φ | N₁) = |⟨n | ψ_{N₁}(φ)⟩|² / ⟨ψ_{N₁}|ψ_{N₁}⟩,
```

and the detected distribution is again the binomial-loss image (★) with an added
Poisson background.

### 3.1 Closed form

Insert a Fock resolution between the two operations:

```
 ⟨n|ψ_{N₁}(φ)⟩ = Σ_m  ⟨n| S(r₂) |m⟩ · e^{−iφ m} · ⟨m| a^{N₁} S(r₁) |0⟩ .
```

Two known ingredients:

* **Squeezed-vacuum amplitudes** (input to the tap):
  `⟨k| S(r₁)|0⟩ = c_k(r₁)`, with `c_{2j}(r₁) = (−tanh r₁)^j √((2j)!) / (2^j j! √cosh r₁)`
  and `c_odd = 0`. Hence
  `⟨m| a^{N₁} S(r₁)|0⟩ = √((m+N₁)!/m!) · c_{m+N₁}(r₁)` (non-zero only when `m+N₁` is even).

* **Single-mode squeezing matrix element** `⟨n|S(r₂)|m⟩` (real `r₂`, non-zero for `n−m`
  even) has the standard closed form in terms of a hypergeometric/Hermite polynomial,

  ```
  ⟨n|S(r)|m⟩ = √(n! m!) (cosh r)^{-1/2} Σ_j
        (−½ tanh r)^{(n-... )}  …   (Gradshteyn–Ryzhik 3.462; e.g. Møller 1996).
  ```

Putting it together,

```
 P_id(n,φ|N₁) ∝ | Σ_m ⟨n|S(r₂)|m⟩ e^{−iφ m} √((m+N₁)!/m!) c_{m+N₁}(r₁) |² ,   (m+N₁ even)
```

a **finite sum of elementary/hypergeometric terms** — the exact analytical answer.
For `N₁ ≥ 1` the state is **non-Gaussian** (photon-subtracted squeezed vacuum,
then re-squeezed), so unlike §2 it is *not* a single squeezed-vacuum with an effective
`s(φ)`; the full sum must be kept. `ni_analysis.py` evaluates it exactly in a truncated
Fock space (identical numbers to the closed form, to machine precision).

### 3.2 Physical effect

Subtracting a photon from squeezed vacuum **removes the vacuum component and raises the
mean photon number** (`⟨n⟩` for one-photon subtraction is `≈ 3 sinh²r₁` for weak
squeezing, versus `sinh²r₁` without subtraction). In the data this shows up directly:
`⟨n⟩_det` jumps from ≈0.015 (N₁=0) to ≈0.10 (N₁=1). The subtracted state is more
non-classical (negative Wigner function for `N₁` odd) but, as seen in the data, its
**fringe visibility can be lower** — whether subtraction helps the phase sensitivity is
exactly the question the CFI answers below.

---

## 3b. Loss anywhere in the interferometer (internal loss + separate detectors)

Loss at the **end** (detector) is a simple binomial thinning of the final distribution,
Eq. (★). Loss **inside** the interferometer — after the first squeezer, at the herald tap,
or before the second squeezer — is *not*, because the second squeezer **re-amplifies**
whatever leaks in. A photon lost before `S₂` is replaced by vacuum that `S₂` turns into
*thermal* photons, so internal loss raises the higher photon-number sectors
(`P(2)`, `P(3)`) rather than just rescaling. It must therefore be applied as a proper
loss channel on the **density matrix**, between the operators:

```
 ρ = |0⟩⟨0|
 ρ → S₁ ρ S₁†
 ρ → Λ_{η_int}[ρ]                 # internal loss (after S1 / at the herald tap)
 ρ → a^{N₁} ρ a†^{N₁} / tr(…)     # heralded photon subtraction
 ρ → S₂ R(φ) ρ R(φ)† S₂†          # phase + second squeezer
 P_d(n,φ) = binom-loss_{η_d}[ diag ρ ] ⊛ Poisson(n_bg)   for d = 3, 4
```

where `Λ_η[ρ]_{j,k} = Σ_l √(C(j+l,l) C(k+l,l)) (1−η)^l η^{(j+k)/2} ρ_{j+l,k+l}` is the
amplitude-damping (loss) channel. The two detectors 3 and 4 (the 50:50 split of the
output) get **independent efficiencies η₃, η₄**; both marginals come from the *same*
output density matrix. This is implemented in `ni_analysis.py`
(`loss_channel`, `output_diag`, `apply_detection`, `fit_pooled_general`) and reduces
exactly to Eq. (★) when `η_int = 1` (verified to 1e-15).

**Loss in front of the herald arm** is different again: it lowers the herald *efficiency*.
Its leading effect is on herald *rates* and a slight blurring of "exactly N₁ subtracted";
for the conditional output *shape* in the weak-tap limit it is second order, so it is not
included as a free parameter here (it can be added as a herald POVM if needed).

**What the data can and cannot constrain (see RESULTS.md):** η₃ and η₄ are separately
identifiable (their ratio is ~1.2, stable across herald conditions). The **internal loss
η_int is degenerate** with the gain ratio `r₁/r₂` in a single-detector marginal — both
lower the fringe visibility — so a marginal scan cannot fix it (χ² is nearly flat in
η_int). Breaking the degeneracy needs either single-pass calibration of the gains/
efficiencies (as in the reference, Appendix F) or the full 2-D joint `P(n₃,n₄)`.

## 4. Fisher information, sensitivity, and the classical benchmark

For a measured photon-number distribution `p_n(φ)` the **classical Fisher information** is

```
 F(φ) = Σ_n (1/p_n(φ)) (∂p_n(φ)/∂φ)² .
```

The single-shot phase sensitivity is `Δφ = 1/√F(φ)`; for `M` independent shots
`Δφ = 1/√(M F)`. As in the reference we take `p_n` **from the measured data** and the
derivative `∂p_n/∂φ` **from the validated model**, which avoids amplifying counting
noise by numerical differentiation.

Two benchmarks are computed for comparison:

* **Click detection** (what a bucket/threshold detector would see): collapse to
  `{0, ≥1}`, `p_click = 1 − p_0`, and `F_click = (∂p_0/∂φ)² / [p_0(1−p_0)]`.
  The ratio `F_PNR / F_click` is the *photon-number-resolution advantage* and is
  independent of the overall efficiency scale — the most robust number to quote.

* **Shot-noise limit (classical / SNL).** The classical reference is a coherent state
  carrying the same number of photons that probe the phase *inside* the interferometer,
  i.e. the mean photon number after the first squeezer, `N_in = sinh²(r₁)` (single mode).
  For a coherent state the maximum CFI equals the photon number, so

  ```
   F_SNL = sinh²(r₁),      Δφ_SNL = 1/ sinh(r₁).
  ```

  (This is the single-mode version of the reference's `F_SNL = 2 sinh²G₁`, which counts
  the two signal+idler photons.) `sinh²(r₁)` is obtained two ways in the code: from the
  joint fit, and independently from the herald-arm statistics via
  `g⁽²⁾ = 3 + 1/sinh²r₁` for single-mode squeezed vacuum — a **loss-independent**
  estimate, because `g⁽²⁾` is invariant under linear loss.

A measured `F(φ) > F_SNL` over some phase range is an (internal) sub-shot-noise /
quantum-enhanced result; `F_PNR > F_click` quantifies what is gained by counting photons
rather than just detecting "a click".

### Phase convention
`φ` is the single-mode optical phase in `R(φ)=e^{−iφ n}`; the ⟨n⟩ fringe therefore has
period `π` in `φ`. All CFI and `Δφ` values are per radian of this `φ`. Because the
PNR-vs-click ratio is dimensionless it is unaffected by this choice; only the absolute
`Δφ` carries the convention.
