"""
ni_joint.py  -- full 2-D joint P(n3,n4) model + multimode/thermal background,
building on the single-mode primitives in ni_analysis.py.

Key physics (single output mode split 50:50 onto detectors 3 & 4):
  * number-resolved detection with loss is a number-DIAGONAL POVM, and a 50:50 BS
    conserves photon number, so P(n3,n4) depends only on the OUTPUT diagonal d_out(m):
        P(n3,n4) = sum_m d_out(m) * Trinomial(n3,n4 | m ; p3, p4),   p_d = 0.5*eta_d
    which factorises as
        P(n3,n4) = Ptot(s=n3+n4) * C(s,n3) r^n3 (1-r)^n4 ,  r = p3/(p3+p4),
        Ptot = binomial-loss(d_out, p3+p4).
  * => the split ratio r is phase-INDEPENDENT: the total s=n3+n4 is a sufficient
    statistic for the phase. The joint therefore carries exactly the same phase
    (Fisher) information as the total photon number; the 2-D structure only calibrates
    eta3/eta4. (Verified numerically in cfi_check().)
Multimode/thermal background: extra super-Poissonian photons in the output mode,
  d_out -> d_out (*) thermal(n_th), lift P(2)/P(3) relative to a single-mode + Poisson
  model and remove the N1=1,2 residuals.
"""
import os, math, numpy as np
from scipy.optimize import least_squares
from scipy.special import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ni_analysis as A

np.seterr(all="ignore")
FIG_DIR = A.FIG_DIR


def thermal_dist(n_th, D):
    k = np.arange(D)
    return (n_th ** k) / ((1 + n_th) ** (k + 1))


def poisson_dist(mu, D):
    k = np.arange(D)
    return np.exp(-mu) * mu ** k / np.array([math.factorial(i) for i in k])


def joint_model(diag_out, eta3, eta4, nmax, D, n_bg=0.0, bg="poisson"):
    """P(n3,n4) [nmax+1 x nmax+1] from an output diagonal, with a background in the
    output mode (thermal or Poisson) added BEFORE the 50:50 split."""
    d = diag_out.copy()
    if n_bg > 1e-9:
        b = thermal_dist(n_bg, D) if bg == "thermal" else poisson_dist(n_bg, D)
        d = np.convolve(d, b)[:D]
    p3, p4 = 0.5 * eta3, 0.5 * eta4
    ps = p3 + p4
    r = p3 / ps if ps > 0 else 0.5
    # total-photon distribution after combined loss ps
    Ls = A._loss_matrix(ps, D)                 # Ls[s,m]=C(m,s) ps^s (1-ps)^(m-s)
    Ptot = Ls @ d                              # [D]
    P = np.zeros((nmax + 1, nmax + 1))
    for n3 in range(nmax + 1):
        for n4 in range(nmax + 1):
            s = n3 + n4
            if s < D:
                P[n3, n4] = Ptot[s] * comb(s, n3) * r ** n3 * (1 - r) ** n4
    return P


# ---------------------------------------------------------------- data
def load_joint(direction, nsubs, nmax=4):
    """Per round/condition: V, list of 2-D matrices (n3 x n4 counts, truncated to nmax), N."""
    vfolders = A.voltage_folders(direction)
    import glob, re
    rounds = {}
    for v, fp in vfolders:
        for rp in glob.glob(os.path.join(fp, "*_heralded_3way.txt")):
            k = int(re.search(r"round_(\d+)_", os.path.basename(rp)).group(1))
            rounds.setdefault(k, {})
    data = {k: {ns: {"V": [], "M": []} for ns in nsubs} for k in rounds}
    for v, fp in vfolders:
        for rp in glob.glob(os.path.join(fp, "*_heralded_3way.txt")):
            k = int(re.search(r"round_(\d+)_", os.path.basename(rp)).group(1))
            mats, _ = A.parse_heralded_file(rp)
            for ns in nsubs:
                if ns in mats:
                    m = mats[ns]                       # rows=Det3, cols=Det4
                    mm = np.zeros((nmax + 1, nmax + 1))
                    r0 = min(nmax + 1, m.shape[0]); c0 = min(nmax + 1, m.shape[1])
                    mm[:r0, :c0] = m[:r0, :c0]
                    data[k][ns]["V"].append(v); data[k][ns]["M"].append(mm)
    out = {}
    for k in sorted(rounds):
        entry = {}; ok = True
        for ns in nsubs:
            V = np.array(data[k][ns]["V"])
            if len(V) < 6:
                ok = False; break
            idx = np.argsort(V)
            entry[ns] = {"V": V[idx], "M": [data[k][ns]["M"][j] for j in idx],
                         "N": np.array([data[k][ns]["M"][j].sum() for j in idx])}
        if ok:
            out[k] = entry
    return out, sorted(out)


def pool_joint(rounds, rids, nsubs, omega, phi0_list, nbins=40):
    """Fold to phi in [0,pi), bin, sum 2-D matrices per bin."""
    pooled = {}
    edges = np.linspace(0, np.pi, nbins + 1)
    for ns in nsubs:
        acc = [None] * nbins; Nc = np.zeros(nbins)
        for k in rids:
            if ns not in rounds[k]:
                continue
            phi = np.mod(omega * rounds[k][ns]["V"] + phi0_list[k], np.pi)
            b = np.clip(np.digitize(phi, edges) - 1, 0, nbins - 1)
            for j, bb in enumerate(b):
                M = rounds[k][ns]["M"][j]
                acc[bb] = M.copy() if acc[bb] is None else acc[bb] + M
                Nc[bb] += rounds[k][ns]["N"][j]
        ctr = 0.5 * (edges[:-1] + edges[1:])
        keep = [i for i in range(nbins) if acc[i] is not None and Nc[i] > 0]
        pooled[ns] = {"phi": ctr[keep], "M": [acc[i] for i in keep], "N": Nc[np.array(keep)]}
    return pooled


# ---------------------------------------------------------------- fit
def fit_joint(pooled, ns, r1, D=16, bg="poisson", sysfloor=0.02, fit_eta_int=True):
    """Fit the 2-D joint of ONE condition: r2, [eta_int], eta3, eta4, n_bg, delta."""
    phi = pooled[ns]["phi"]; N = pooled[ns]["N"]
    Mobs = np.array([m / n for m, n in zip(pooled[ns]["M"], N)])   # [nb, K, K]
    nmax = Mobs.shape[1] - 1
    sig = np.sqrt(np.maximum(Mobs * (1 - Mobs), 1e-12) / N[:, None, None])
    sig = np.sqrt(sig ** 2 + (sysfloor * np.maximum(Mobs, 1e-5)) ** 2)
    sig = np.maximum(sig, 1e-8)

    def unpack(p):
        i = 0
        r2 = p[i]; i += 1
        eta_int = p[i] if fit_eta_int else 1.0; i += 1 if fit_eta_int else 0
        eta3, eta4, nbg, delta = p[i], p[i + 1], p[i + 2], p[i + 3]
        return r2, eta_int, eta3, eta4, nbg, delta

    def resid(p):
        r2, eta_int, eta3, eta4, nbg, delta = unpack(p)
        diag = A.output_diag(phi + delta, r1, r2, eta_int, ns, D)
        out = []
        for i in range(len(phi)):
            M = joint_model(diag[i], eta3, eta4, nmax, D, nbg, bg)
            out.append(((Mobs[i] - M) / sig[i]).ravel())
        return np.concatenate(out)

    if fit_eta_int:
        lb = [0.005, 0.05, 1e-3, 1e-3, 0, -np.pi]; ub = [3, 1.0, 1.0, 1.0, 0.5, np.pi]
        p0s = [[0.1, 1.0, 0.08, 0.08, 0.003, d] for d in (0, np.pi / 2, np.pi, -np.pi / 2)]
    else:
        lb = [0.005, 1e-3, 1e-3, 0, -np.pi]; ub = [3, 1.0, 1.0, 0.5, np.pi]
        p0s = [[0.1, 0.08, 0.08, 0.003, d] for d in (0, np.pi / 2, np.pi, -np.pi / 2)]
    best = None
    for p0 in p0s:
        try:
            r = least_squares(resid, np.clip(p0, lb, ub), bounds=(lb, ub),
                              method="trf", max_nfev=3000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    r2, eta_int, eta3, eta4, nbg, delta = unpack(best.x)
    ndof = max(Mobs.size - len(best.x), 1)
    return {"r1": r1, "r2": r2, "eta_int": eta_int, "eta3": eta3, "eta4": eta4,
            "nbg": nbg, "delta": delta, "bg": bg, "chi2": 2 * best.cost / ndof}


# ---------------------------------------------------------------- CFI: joint vs total vs marginal
def cfi_check(par, ns, r1, D=16, ngrid=200):
    """Peak CFI from (i) full 2-D joint, (ii) total n3+n4, (iii) single Det4 marginal."""
    phi = np.linspace(0, np.pi, ngrid); dphi = 1e-4
    def joints(ph):
        diag = A.output_diag(ph, r1, par["r2"], par["eta_int"], ns, D)
        return np.array([joint_model(diag[i], par["eta3"], par["eta4"], 6, D,
                                     par["nbg"], par["bg"]) for i in range(len(ph))])
    J = joints(phi + par["delta"]); Jp = joints(phi + par["delta"] + dphi)
    dJ = (Jp - J) / dphi
    F_joint = np.nansum(dJ ** 2 / (J + 1e-15), axis=(1, 2))
    # total s = n3+n4
    def tot(Jarr):
        K = Jarr.shape[1]; T = np.zeros((Jarr.shape[0], 2 * K - 1))
        for a in range(K):
            for b in range(K):
                T[:, a + b] += Jarr[:, a, b]
        return T
    T, Tp = tot(J), tot(Jp); dT = (Tp - T) / dphi
    F_tot = np.nansum(dT ** 2 / (T + 1e-15), axis=1)
    # Det4 marginal
    m4 = J.sum(1); m4p = Jp.sum(1); dm4 = (m4p - m4) / dphi
    F_m4 = np.nansum(dm4 ** 2 / (m4 + 1e-15), axis=1)
    return F_joint.max(), F_tot.max(), F_m4.max()
