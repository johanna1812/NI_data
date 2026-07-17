"""
ni_twomode.py -- general TWO-MODE model of the nonlinear interferometer with a
tunable output beam splitter (HWP+PBS), covering BOTH read-out configurations:

    theta_bs = 0       -> PBS separation  (D3, D4 = the two polarisation modes a, b)
    theta_bs = pi/4    -> 50:50           (HWP at 22.5 deg mixes a, b before the PBS)

and both squeezer types:
    sq = 'single'  -> single-mode squeezers on each mode:  S_a(r) (x) S_b(r)
    sq = 'twomode' -> two-mode (parametric) squeezer:       exp(r(ab - a!b!))

Circuit (two polarisation modes a, b; vacuum in):
    |0,0> -> Sq1 -> [subtract N1 photons from chosen mode] -> R(phi) -> Sq2
          -> BeamSplitter(theta_bs) -> per-mode detection loss (eta3, eta4) + bkg
    P(n3,n4) = detected joint photon-number distribution.

Because number-resolved detection with loss is number-diagonal per output mode, the
detected joint is  P = L3 . |psi(a,b)|^2 . L4^T  (per-mode binomial loss) plus an
independent background on each detector. Internal loss is neglected here (set it via
the density-matrix variant if needed); it was found degenerate in the marginals.

Which configuration matches a data set is decided empirically with the Cauchy-Schwarz
ratio R = g11^2/(g2_3 g2_4):  R>1 (pair-correlated) vs R~1 (uncorrelated).  See
identify_config().
"""
import math, numpy as np
from scipy.linalg import expm
from scipy.special import comb
import ni_analysis as A

np.seterr(all="ignore")


def _ops(D):
    ann = np.diag(np.sqrt(np.arange(1, D)), 1)
    I = np.eye(D)
    return np.kron(ann, I), np.kron(I, ann)          # a, b on D*D space


def _loss_matrix(eta, D):
    m = np.arange(D)[None, :]; n = np.arange(D)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        L = comb(m, n) * (eta ** n) * ((1 - eta) ** (m - n))
    L[n > m] = 0.0
    return L


def _bg_vec(mu, D, kind):
    k = np.arange(D)
    if mu <= 1e-9:
        v = np.zeros(D); v[0] = 1.0; return v
    if kind == "thermal":
        return (mu ** k) / ((1 + mu) ** (k + 1))
    return np.exp(-mu) * mu ** k / np.array([math.factorial(i) for i in k])


def two_mode_pn(phi_arr, params, N1=0, nmax=4, D=9):
    """Detected P(n3,n4) vs phi. params keys:
       r1,r2, sq('single'|'twomode'), theta_bs, sub('a'|'b'|'diag'),
       eta3,eta4, nbg3,nbg4, bg('poisson'|'thermal'), phi0."""
    a, b = _ops(D)
    ad, bd = a.T, b.T
    r1, r2 = params["r1"], params["r2"]
    sq = params.get("sq", "single")
    if sq == "single":
        g1 = 0.5 * r1 * (a @ a - ad @ ad) + 0.5 * r1 * (b @ b - bd @ bd)
        g2 = 0.5 * r2 * (a @ a - ad @ ad) + 0.5 * r2 * (b @ b - bd @ bd)
        S1, S2 = expm(g1), expm(g2)
    elif sq == "twomode":                            # two-mode (parametric) squeezers
        S1 = expm(r1 * (a @ b - ad @ bd))
        S2 = expm(r2 * (a @ b - ad @ bd))
    elif sq == "rotated":                            # single-mode squeezer on
        al = params.get("alpha", math.pi / 4)        # mode c = cos(alpha) a + sin(alpha) b
        ca, sa = math.cos(al), math.sin(al)
        c = ca * a + sa * b; cd = c.T
        S1 = expm(0.5 * r1 * (c @ c - cd @ cd))
        S2 = expm(0.5 * r2 * (c @ c - cd @ cd))
    else:                                            # 'general': two-mode + single-mode.
        rt = params.get("rt", r1)                    # r1,r2 = two-mode gains; rs = single-mode
        rt2 = params.get("rt2", r2)
        rs1 = params.get("rs1", 0.0); rs2 = params.get("rs2", 0.0)
        def gen(rt_, rs_):
            return (rt_ * (a @ b - ad @ bd)
                    + 0.5 * rs_ * (a @ a - ad @ ad) + 0.5 * rs_ * (b @ b - bd @ bd))
        S1 = expm(gen(r1, rs1)); S2 = expm(gen(r2, rs2))
    B = expm(params.get("theta_bs", 0.0) * (ad @ b - a @ bd))

    psi1 = S1[:, 0].copy()                            # Sq1 |0,0>
    if N1 > 0:
        sub = params.get("sub", "a")
        op = a if sub == "a" else (b if sub == "b" else (a + b) / math.sqrt(2))
        for _ in range(N1):
            psi1 = op @ psi1
        nrm = np.linalg.norm(psi1)
        if nrm <= 0:
            return np.zeros((len(phi_arr), nmax + 1, nmax + 1))
        psi1 /= nrm

    na = np.arange(D).repeat(D)                       # <n_a> index on the D*D basis
    L3 = _loss_matrix(params.get("eta3", 1.0), D)
    L4 = _loss_matrix(params.get("eta4", 1.0), D)
    b3 = _bg_vec(params.get("nbg3", 0.0), D, params.get("bg", "poisson"))
    b4 = _bg_vec(params.get("nbg4", 0.0), D, params.get("bg", "poisson"))
    phi0 = params.get("phi0", 0.0)

    out = np.empty((len(phi_arr), nmax + 1, nmax + 1))
    for i, phi in enumerate(phi_arr):
        psi = B @ (S2 @ (np.exp(-1j * (phi + phi0) * na) * psi1))
        Pid = (np.abs(psi) ** 2).reshape(D, D)        # |<m_a,m_b|psi>|^2
        Pdet = L3 @ Pid @ L4.T                        # per-mode detection loss
        # independent background per detector
        Pdet = np.array([np.convolve(row, b4)[:D] for row in Pdet])
        Pdet = np.array([np.convolve(col, b3)[:D] for col in Pdet.T]).T
        out[i] = Pdet[:nmax + 1, :nmax + 1]
    return out


# ---- presets for the two read-outs -----------------------------------------
def preset(config, **kw):
    p = dict(r1=0.33, r2=0.05, sq="single", sub="a",
             eta3=0.12, eta4=0.13, nbg3=0.005, nbg4=0.005, bg="poisson", phi0=0.0)
    p["theta_bs"] = 0.0 if config.upper() == "PBS" else math.pi / 4
    p.update(kw)
    return p


# ---- correlation diagnostics ------------------------------------------------
def moments(P):
    n = np.arange(P.shape[0])
    p3, p4 = P.sum(1), P.sum(0)
    m3, m4 = (p3 * n).sum(), (p4 * n).sum()
    g23 = (p3 * n * (n - 1)).sum() / m3 ** 2
    g24 = (p4 * n * (n - 1)).sum() / m4 ** 2
    g11 = (P * np.outer(n, n)).sum() / (m3 * m4)
    return g23, g24, g11, g11 ** 2 / (g23 * g24)


def cs_ratio(params, N1=0, D=9, ngrid=12):
    """Phase-averaged Cauchy-Schwarz ratio R the model predicts for a configuration."""
    phi = np.linspace(0, np.pi, ngrid)
    P = two_mode_pn(phi, params, N1=N1, nmax=6, D=D).sum(0)
    P = P / P.sum()
    return moments(P)[3]


# ---- fitter for two-mode joint data -----------------------------------------
from scipy.optimize import least_squares

def fit_twomode(pooled_ns, N1, sq="twomode", theta_bs=0.0, D=9, sysfloor=0.02):
    """Fit r1,r2,eta3,eta4,nbg3,nbg4,phi0 to one condition's binned pooled joint matrices.
       pooled_ns: dict with 'phi'(nb,), 'M'(list of matrices), 'N'(nb,)."""
    phi = pooled_ns["phi"]; N = pooled_ns["N"]
    Mobs = np.array([m / n for m, n in zip(pooled_ns["M"], N)])
    nmax = Mobs.shape[1] - 1
    sig = np.sqrt(np.maximum(Mobs * (1 - Mobs), 1e-12) / N[:, None, None])
    sig = np.maximum(np.sqrt(sig ** 2 + (sysfloor * np.maximum(Mobs, 1e-5)) ** 2), 1e-8)

    fit_alpha = (sq == "rotated")

    def resid(p):
        par = dict(r1=p[0], r2=p[1], eta3=p[2], eta4=p[3], nbg3=p[4], nbg4=p[5],
                   phi0=p[6], sq=sq, theta_bs=theta_bs, sub="a")
        if fit_alpha:
            par["alpha"] = p[7]
        M = two_mode_pn(phi, par, N1=N1, nmax=nmax, D=D)
        return ((Mobs - M) / sig).ravel()

    lb = [0.02, 0.005, 1e-3, 1e-3, 0, 0, -np.pi]
    ub = [2.0, 2.0, 1.0, 1.0, 0.3, 0.3, np.pi]
    p0b = [0.4, 0.1, 0.12, 0.13, 0.004, 0.004, None]
    if fit_alpha:
        lb += [0.0]; ub += [math.pi / 2]; p0b += [math.pi / 4]
    best = None
    for d0 in (0, np.pi / 2, np.pi, -np.pi / 2):
        p0 = list(p0b); p0[6] = d0
        try:
            r = least_squares(resid, np.clip(p0, lb, ub), bounds=(lb, ub),
                              method="trf", max_nfev=2000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    x = best.x
    out = dict(r1=x[0], r2=x[1], eta3=x[2], eta4=x[3], nbg3=x[4], nbg4=x[5],
               phi0=x[6], sq=sq, theta_bs=theta_bs, sub="a",
               chi2=2 * best.cost / max(Mobs.size - len(x), 1))
    if fit_alpha:
        out["alpha"] = x[7]
    return out


def fit_general(pooled_ns, N1, theta_bs=0.0, D=8, sysfloor=0.02, seeds=None):
    """Fit the GENERAL two-mode squeezer (two-mode gain r1,r2 + single-mode rs1) to one
    condition's binned pooled joint matrices. Returns params dict incl. chi2.
    This is the model that fits the PBS data (chi2~20, eta~0.5 as in the reference)."""
    phi = pooled_ns["phi"]; N = pooled_ns["N"]
    Mobs = np.array([m / n for m, n in zip(pooled_ns["M"], N)])
    nmax = Mobs.shape[1] - 1
    sig = np.maximum(np.sqrt(np.maximum(Mobs * (1 - Mobs), 1e-12) / N[:, None, None]
                             + (sysfloor * np.maximum(Mobs, 1e-5)) ** 2), 1e-8)

    def resid(p):
        par = dict(r1=p[0], rs1=p[1], r2=p[2], rs2=0.0, eta3=p[3], eta4=p[4],
                   nbg3=p[5], nbg4=p[5], phi0=p[6], sq="general", theta_bs=theta_bs)
        return ((Mobs - two_mode_pn(phi, par, N1=N1, nmax=nmax, D=D)) / sig).ravel()

    lb = [0.02, 0.0, 0.0, 1e-3, 1e-3, 0, -np.pi]
    ub = [1.2, 1.0, 1.0, 1.0, 1.0, 0.1, np.pi]
    if seeds is None:
        seeds = [[0.4, 0.15, 0.05, 0.15, 0.16, 0.003, 1.5],
                 [0.5, 0.05, 0.05, 0.13, 0.14, 0.003, 1.5],
                 [0.2, 0.05, 0.05, 0.45, 0.5, 0.003, 1.5]]
    best = None
    for s in seeds:
        try:
            r = least_squares(resid, np.clip(s, lb, ub), bounds=(lb, ub),
                              method="trf", max_nfev=1500)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    x = best.x
    return dict(r1=x[0], rs1=x[1], r2=x[2], rs2=0.0, eta3=x[3], eta4=x[4],
               nbg3=x[5], nbg4=x[5], phi0=x[6], sq="general", theta_bs=theta_bs,
               chi2=2 * best.cost / max(Mobs.size - len(x), 1))


def cfi_joint(params, N1=0, D=9, ngrid=200):
    """Peak classical Fisher info per shot from the TRUE two-mode joint P(n3,n4),
    vs the total n3+n4, vs one detector marginal. For a correlated two-mode state
    F_joint > F_total (correlations carry phase info); for a passive split they are equal."""
    phi = np.linspace(0, np.pi, ngrid); dphi = 1e-4
    J0 = two_mode_pn(phi, params, N1=N1, nmax=6, D=D)
    J1 = two_mode_pn(phi + dphi, params, N1=N1, nmax=6, D=D)
    dJ = (J1 - J0) / dphi
    F_joint = np.nansum(dJ ** 2 / (J0 + 1e-15), axis=(1, 2))
    K = J0.shape[1]
    def tot(x):
        T = np.zeros((x.shape[0], 2 * K - 1))
        for a in range(K):
            for b in range(K):
                T[:, a + b] += x[:, a, b]
        return T
    T0, T1 = tot(J0), tot(J1); dT = (T1 - T0) / dphi
    F_tot = np.nansum(dT ** 2 / (T0 + 1e-15), axis=1)
    m0, m1 = J0.sum(2), J1.sum(2); dm = (m1 - m0) / dphi          # Det3 marginal
    F_m = np.nansum(dm ** 2 / (m0 + 1e-15), axis=1)
    return dict(F_joint=F_joint.max(), F_total=F_tot.max(), F_marg=F_m.max(),
                gain_joint_over_total=F_joint.max() / F_tot.max())
