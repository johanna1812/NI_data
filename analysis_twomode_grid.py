"""
analysis_twomode_grid.py
Full two-mode (PBS) analysis:
  * fit the general two-mode source + detector efficiencies on N1=0 (no subtraction);
  * for N1=1,2, FIX the detector efficiencies (same detectors!) and the source gains,
    and test which subtraction mode ('a','b','diag') the heralded data prefer;
  * CFI per shot from the TRUE two-mode joint vs total(n3+n4) vs one detector;
  * figures.
Run:  python3 analysis_twomode_grid.py
"""
import math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ni_analysis as A
import ni_joint as J
import ni_twomode as T
from scipy.optimize import least_squares

np.seterr(all="ignore")


def _prep(direction, nsubs):
    rounds, rids = J.load_joint(direction, nsubs)
    oml, phi0 = [], {}
    for k in rids:
        om, ph, _ = A.prefit_omega_phi0(
            rounds[k][0]["V"], np.array([m.sum(0) for m in rounds[k][0]["M"]]),
            rounds[k][0]["N"])
        oml.append(om); phi0[k] = ph
    omega = float(np.median(oml))
    pooled = J.pool_joint(rounds, rids, nsubs, omega, phi0, nbins=12)
    return pooled, omega


def _fit_subtracted(pooled_ns, N1, base, D=8):
    """Fix source gains + efficiencies from the N1=0 fit; fit only phi0, per-condition
    background, and try each subtraction mode. Returns best (params, chi2, submode)."""
    phi = pooled_ns["phi"]; N = pooled_ns["N"]
    Mobs = np.array([m / n for m, n in zip(pooled_ns["M"], N)])
    nmax = Mobs.shape[1] - 1
    sig = np.maximum(np.sqrt(np.maximum(Mobs * (1 - Mobs), 1e-12) / N[:, None, None]
                            + (0.02 * np.maximum(Mobs, 1e-5)) ** 2), 1e-8)
    best = None
    for sub in ("a", "b", "diag"):
        def resid(p):
            par = dict(base); par.update(phi0=p[0], nbg3=p[1], nbg4=p[1], sub=sub)
            return ((Mobs - T.two_mode_pn(phi, par, N1=N1, nmax=nmax, D=D)) / sig).ravel()
        for d0 in (base["phi0"], 0.0, math.pi):
            try:
                r = least_squares(resid, [d0, 0.003], bounds=([-np.pi, 0], [np.pi, 0.1]),
                                  method="trf", max_nfev=800)
                chi2 = 2 * r.cost / max(Mobs.size - 2, 1)
                if best is None or chi2 < best[1]:
                    par = dict(base); par.update(phi0=r.x[0], nbg3=r.x[1], nbg4=r.x[1], sub=sub)
                    best = (par, chi2, sub)
            except Exception:
                pass
    return best


def run(direction="UP", nsubs=(0, 1, 2), D=8):
    pooled, omega = _prep(direction, nsubs)
    print(f"\n=== TWO-MODE (PBS) grid [{direction}] ===")

    p0 = T.fit_general(pooled[0], N1=0, theta_bs=0.0, D=D)
    print(f"N1=0 (no subtraction): chi2={p0['chi2']:.0f}  "
          f"two-mode gain rt={p0['r1']:.3f}, single-mode rs={p0['rs1']:.3f}, r2={p0['r2']:.3f}")
    print(f"          detectors: eta3={p0['eta3']:.3f}, eta4={p0['eta4']:.3f}  "
          f"(cf. reference TES 0.54/0.58)")

    base = dict(sq="general", theta_bs=0.0, r1=p0["r1"], rs1=p0["rs1"], r2=p0["r2"],
                rs2=0.0, eta3=p0["eta3"], eta4=p0["eta4"], phi0=p0["phi0"])
    params = {0: p0}
    for ns in nsubs:
        if ns == 0:
            continue
        par, chi2, sub = _fit_subtracted(pooled[ns], ns, base, D=D)
        params[ns] = par
        print(f"N1={ns} (subtract, fixed detectors): chi2={chi2:.0f}  best subtraction mode='{sub}'")

    print("\nCFI per shot from the TRUE two-mode joint vs total vs one detector:")
    rows = {}
    for ns in nsubs:
        c = T.cfi_joint(params[ns], N1=ns, D=max(D, 9))
        rows[ns] = c
        print(f"  N1={ns}: F_joint={c['F_joint']:.5f}  F_total={c['F_total']:.5f}  "
              f"F_1det={c['F_marg']:.5f}   joint>total by {100*(c['gain_joint_over_total']-1):+.0f}%")

    # figure: joint fit quality (data vs model heatmap) for N1=0, and CFI bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    Pd = sum(pooled[0]["M"]); Pd = Pd / Pd.sum()
    Pm = T.two_mode_pn(pooled[0]["phi"], p0, N1=0, nmax=Pd.shape[0] - 1, D=D)
    Pm = sum(Pm[i] * pooled[0]["N"][i] for i in range(len(pooled[0]["phi"]))); Pm /= Pm.sum()
    K = min(5, Pd.shape[0])
    for ax, M, ttl in ((axes[0], Pd, "data P(n3,n4)"), (axes[1], Pm, "two-mode model")):
        im = ax.imshow(np.log10(M[:K, :K] + 1e-9), origin="lower", cmap="viridis",
                       vmin=-7, vmax=-0.5)
        ax.set_title(ttl); ax.set_xlabel("n4 (Det4)"); ax.set_ylabel("n3 (Det3)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="log10 P")
    x = np.arange(len(nsubs))
    axes[2].bar(x - 0.2, [rows[n]["F_joint"] for n in nsubs], 0.2, label="joint")
    axes[2].bar(x, [rows[n]["F_total"] for n in nsubs], 0.2, label="total n3+n4")
    axes[2].bar(x + 0.2, [rows[n]["F_marg"] for n in nsubs], 0.2, label="1 detector")
    axes[2].set_yscale("log"); axes[2].set_xticks(x)
    axes[2].set_xticklabels(["no subtr" if n == 0 else f"N1={n}" for n in nsubs])
    axes[2].set_ylabel("peak CFI per shot"); axes[2].set_title("CFI: joint vs total vs 1 det")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Two-mode (PBS) analysis [{direction}]: general squeezer, shared detectors",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{A.FIG_DIR}/fig10_twomode_grid_{direction}.png", dpi=130)
    plt.close(fig)
    print(f"  fig10 written to {A.FIG_DIR}/")
    return params, rows


if __name__ == "__main__":
    run("UP")
