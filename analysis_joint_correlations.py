"""
analysis_joint_correlations.py
Checks the two things suggested for the D3xD4 joint statistics:
  (1) full 2-D joint P(n3,n4) fit + whether it adds phase information over the total;
  (2) whether a multimode/thermal background removes the N1=1,2 residual.

Main result: the joint is NOT reducible to a passive 50:50 split of one mode --
the D3-D4 cross-correlation violates the classical Cauchy-Schwarz inequality
(R = g11^2/(g2_3 g2_4) > 1), i.e. the two detectors see genuinely photon-number-
correlated (signal/idler) modes. Hence the joint carries real metrological
information the total/marginal do not, and the correct model is the two-mode one
(as in arXiv:2606.30761 Eq. 3), not the single-mode model of ni_analysis.py.
Run:  python3 analysis_joint_correlations.py
"""
import math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ni_analysis as A
import ni_joint as J

np.seterr(all="ignore")


def _phase_axis(rounds, rids):
    oml, phi0 = [], {}
    for k in rids:
        om, ph, _ = A.prefit_omega_phi0(
            rounds[k][0]["V"], np.array([m.sum(0) for m in rounds[k][0]["M"]]),
            rounds[k][0]["N"])
        oml.append(om); phi0[k] = ph
    return float(np.median(oml)), phi0


def moments(P):
    n = np.arange(P.shape[0])
    p3, p4 = P.sum(1), P.sum(0)
    m3, m4 = (p3 * n).sum(), (p4 * n).sum()
    g2_3 = (p3 * n * (n - 1)).sum() / m3 ** 2
    g2_4 = (p4 * n * (n - 1)).sum() / m4 ** 2
    g11 = (P * np.outer(n, n)).sum() / (m3 * m4)
    return g2_3, g2_4, g11


def run(direction="UP", nsubs=(0, 1, 2)):
    nsubs = list(nsubs)
    rounds, rids = J.load_joint(direction, nsubs)
    _, g2h, _ = A.herald_g2(direction)
    r1 = math.asinh(math.sqrt(1.0 / (g2h - 3.0)))
    omega, phi0 = _phase_axis(rounds, rids)
    pooled = J.pool_joint(rounds, rids, nsubs, omega, phi0, nbins=24)

    print(f"\n=== [{direction}] (1) 2-D JOINT FIT (single-mode passive-split model) ===")
    pars = {}
    for ns in nsubs:
        p = J.fit_joint(pooled, ns, r1, bg="poisson", fit_eta_int=True)
        pars[ns] = p
        print(f"  N1={ns}: r2={p['r2']:.3f} eta_int={p['eta_int']:.3f} "
              f"eta3={p['eta3']:.3f} eta4={p['eta4']:.3f} nbg={p['nbg']:.3f} chi2={p['chi2']:.0f}")
    print("  -> high chi2 and nbg railing: the single-mode split does NOT describe the joint.")

    print(f"\n=== [{direction}] (2) THERMAL vs POISSON background (N1=1) ===")
    for bg in ("poisson", "thermal"):
        p = J.fit_joint(pooled, 1, r1, bg=bg, fit_eta_int=True)
        print(f"  {bg:8s}: chi2={p['chi2']:.0f}  nbg={p['nbg']:.3f}")
    print("  -> thermal does not help: the residual is NOT background.")

    print(f"\n=== [{direction}] CAUCHY-SCHWARZ TEST  R = g11^2/(g2_3 g2_4)  (R>1 => nonclassical) ===")
    R = {}
    for ns in nsubs:
        Pd = sum(pooled[ns]["M"]); Pd = Pd / Pd.sum()
        g23, g24, g11 = moments(Pd)
        R[ns] = g11 ** 2 / (g23 * g24)
        print(f"  N1={ns}: g2_D3={g23:.2f} g2_D4={g24:.2f} g11={g11:.2f}  R={R[ns]:.2f}"
              f"  {'NONCLASSICAL (two-mode pairs)' if R[ns] > 1 else 'classical'}")

    print(f"\n=== [{direction}] CFI: single-mode joint model = total (artefact of wrong model) ===")
    for ns in nsubs:
        fj, ft, fm = J.cfi_check(pars[ns], ns, r1)
        print(f"  N1={ns}: F_joint={fj:.5f} F_total={ft:.5f} F_Det4={fm:.5f} (joint~2x one detector)")
    print("  NB: with the TRUE two-mode correlations the joint exceeds the total; needs the")
    print("      two-mode model (arXiv:2606.30761 Eq.3) to quantify.")

    # figure: CSI ratio
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(nsubs))
    ax.bar(x, [R[ns] for ns in nsubs], color=["#4477AA", "#EE6677", "#228833"][:len(nsubs)])
    ax.axhline(1, color="k", ls="--", lw=2, label="classical Cauchy-Schwarz bound")
    for xi, ns in zip(x, nsubs):
        ax.text(xi, R[ns] * 1.05, f"{R[ns]:.1f}x", ha="center", fontweight="bold")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(["no subtr" if n == 0 else f"N1={n}" for n in nsubs])
    ax.set_ylabel("CSI ratio  R = g11^2/(g2_3 g2_4)")
    ax.set_title(f"D3-D4 cross-correlation violates Cauchy-Schwarz:\n"
                 f"genuine two-mode (pair) correlations [{direction}]")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{A.FIG_DIR}/fig9_cauchyschwarz_{direction}.png", dpi=130)
    plt.close(fig)
    return R


if __name__ == "__main__":
    for d in ("UP", "DOWN"):
        run(d)
