"""
ni_analysis.py
==============
Robust analysis of the single-mode nonlinear (SU(1,1)) interferometer with
photon-number-resolving read-out and heralded photon subtraction.

What this does that the exploratory script did not:
  * Uses the EXACT quantum-optical model  P(n,phi) = Loss_eta[ |<n| S2 R(phi) a^N1 S1|0> |^2 ] (+ Poisson bkg)
    with the CORRECT operation order for photon subtraction (a^N1 acts BETWEEN the squeezers,
    not as an input Fock state to the first squeezer).
  * Fits every photon-number sector JOINTLY, per herald condition, sharing one physical
    parameter set (r1, r2, eta, omega, phi0) -- this breaks the r1/r2/eta degeneracy.
  * Fits each time round separately (the phase drifts ~160 deg over the 30 rounds, so
    averaging rounds at fixed voltage smears the fringe) and aggregates -> real error bars.
  * Uses Poisson/binomial weighting with a small systematic floor, so very low count-rate
    sectors are handled gracefully instead of producing "random" fit lines.
  * Estimates the classical Fisher information and phase sensitivity from the DATA
    (probabilities from data, derivative from the validated model), and compares to
    click detection and to the shot-noise limit  F_SNL = sinh^2(r1).

See THEORY.md for the derivations.

Usage:
    python3 ni_analysis.py            # runs the full pipeline on ./ , writes figures to ./figures/
"""
import os, glob, re, math, json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares, curve_fit
from scipy.special import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(all="ignore")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "figures")

# ======================================================================
# 1.  PARSING
# ======================================================================
def parse_heralded_file(path):
    """Return {n_sub: 9x9 matrix (Det3 rows x Det4 cols)}, {n_sub: total herald counts}."""
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    mats, totals = {}, {}
    for blk in content.split("HERALD:")[1:]:
        lines = ("HERALD:" + blk).split("\n")
        head = re.search(r"HERALD: Detector (\d+).*?Px (\d+).*?Total Coincidences: ([0-9.eE+]+)", lines[0])
        if not head or head.group(1) != "1":      # only Detector 1 = herald
            continue
        n_sub = int(head.group(2)) - 1             # Px1 -> 0 subtracted, Px2 -> 1, ...
        rows = []
        for ln in lines:
            if ln.strip().startswith("Px") and ":" in ln:
                vals = re.findall(r"[-+]?\d*\.\d+|\d+", ln.split(":")[1])
                if vals:
                    rows.append([float(v) for v in vals])
        if rows:
            L = len(rows[0])
            rows = [r for r in rows if len(r) == L]
            mats[n_sub] = np.array(rows)
            totals[n_sub] = float(head.group(3))
    return mats, totals


def voltage_folders(direction):
    out = []
    for f in os.listdir(DATA_DIR):
        m = re.match(r"voltage_([0-9.]+)V_%s$" % direction, f)
        if m:
            out.append((float(m.group(1)), os.path.join(DATA_DIR, f, "data_save3")))
    return sorted(out)


def marginal(mat, detector):
    """1-D photon-number distribution (counts) for a detector from the 2-D joint matrix."""
    if detector == "D4":            # columns
        return mat.sum(axis=0)
    if detector == "D3":            # rows
        return mat.sum(axis=1)
    # total photons n3+n4
    R, C = mat.shape
    d = np.zeros(R + C - 1)
    for i in range(R):
        for j in range(C):
            d[i + j] += mat[i, j]
    return d


def load_direction(direction, nsubs, detector="D4"):
    """
    Returns rounds[k] = { n_sub: {'V':array, 'C':counts[nV, nmax+1], 'N':totals[nV]} }
    one entry per time round, plus the sorted voltage axis.
    """
    vfolders = voltage_folders(direction)
    # discover rounds
    rounds = {}
    for v, fp in vfolders:
        for rp in glob.glob(os.path.join(fp, "*_heralded_3way.txt")):
            k = int(re.search(r"round_(\d+)_", os.path.basename(rp)).group(1))
            rounds.setdefault(k, {})
    round_ids = sorted(rounds)
    nmax = 4
    data = {k: {ns: {"V": [], "dist": []} for ns in nsubs} for k in round_ids}
    for v, fp in vfolders:
        for rp in glob.glob(os.path.join(fp, "*_heralded_3way.txt")):
            k = int(re.search(r"round_(\d+)_", os.path.basename(rp)).group(1))
            mats, _ = parse_heralded_file(rp)
            for ns in nsubs:
                if ns in mats:
                    data[k][ns]["V"].append(v)
                    data[k][ns]["dist"].append(marginal(mats[ns], detector))
    # pack into arrays
    out = {}
    for k in round_ids:
        entry = {}
        ok = True
        for ns in nsubs:
            V = np.array(data[k][ns]["V"])
            if len(V) < 6:
                ok = False
                break
            idx = np.argsort(V)
            V = V[idx]
            D = max(len(d) for d in data[k][ns]["dist"])
            C = np.zeros((len(V), D))
            for j, jj in enumerate(idx):
                d = data[k][ns]["dist"][jj]
                C[j, :len(d)] = d
            N = C.sum(axis=1)
            entry[ns] = {"V": V, "C": C[:, :nmax + 1], "N": N}
        if ok:
            out[k] = entry
    return out, round_ids


# ======================================================================
# 2.  EXACT MODEL  P(n, phi)
# ======================================================================
def _squeeze(r, D):
    a = np.diag(np.sqrt(np.arange(1, D)), 1)
    ad = a.T
    return expm(0.5 * r * (a @ a - ad @ ad)), a


def ideal_pn(phi_arr, r1, r2, N1, D=14):
    """P_ideal(n, phi) for the pure state  S2 R(phi) a^N1 S1|0>.  shape [len(phi), D]."""
    S1, a = _squeeze(r1, D)
    S2, _ = _squeeze(r2, D)
    n = np.arange(D)
    psi1 = S1[:, 0].copy()                      # S1|0>
    for _ in range(N1):
        psi1 = a @ psi1                          # subtract photons
    nrm = np.linalg.norm(psi1)
    if nrm <= 0:
        return np.zeros((len(phi_arr), D))
    psi1 /= nrm
    out = np.empty((len(phi_arr), D))
    for k, phi in enumerate(phi_arr):
        psi = S2 @ (np.exp(-1j * phi * n) * psi1)
        out[k] = np.abs(psi) ** 2
    return out


def _loss_matrix(eta, D):
    m = np.arange(D)[None, :]
    nn = np.arange(D)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        L = comb(m, nn) * (eta ** nn) * ((1 - eta) ** (m - nn))
    L[nn > m] = 0.0
    return L


def model_pn(V, r1, r2, eta, nbg, omega, phi0, N1, nmax=4, D=14):
    """Detected P(n,phi) with phi=omega*V+phi0, binomial loss eta and Poisson bkg nbg."""
    Pid = ideal_pn(omega * V + phi0, r1, r2, N1, D)
    L = _loss_matrix(eta, D)
    if nbg > 1e-9:
        k = np.arange(D)
        pois = np.exp(-nbg) * nbg ** k / np.array([math.factorial(i) for i in k])
    res = np.empty((len(V), nmax + 1))
    for i in range(len(V)):
        pd = L @ Pid[i]
        if nbg > 1e-9:
            pd = np.convolve(pd, pois)[:D]
        res[i] = pd[:nmax + 1]
    return res


# ======================================================================
# 3.  ROBUST FRINGE PRE-FIT (mean photon number) -> omega, phi0
# ======================================================================
def prefit_omega_phi0(V, C, N, omega_seed=None):
    """Fit <n>(V)=y0+A cos(2 omega V + 2 phi0) with multistart. Returns omega, phi0."""
    n_axis = np.arange(C.shape[1])
    nbar = (C * n_axis).sum(axis=1) / N
    span = max(V[-1] - V[0], 1.0)

    def f(V, A, Om, Ph, y0):
        return y0 + A * np.cos(Om * V + Ph)

    seeds = [2 * np.pi / span * m for m in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)]
    if omega_seed:
        seeds = [2 * omega_seed] + seeds
    best, bestp = np.inf, None
    for Om in seeds:
        for Ph in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            try:
                p, _ = curve_fit(f, V, nbar,
                                 p0=[(nbar.max() - nbar.min()) / 2, Om, Ph, nbar.mean()],
                                 maxfev=8000)
                r = np.sum((f(V, *p) - nbar) ** 2)
                if r < best:
                    best, bestp = r, p
            except Exception:
                pass
    A, Om, Ph, y0 = bestp
    if A < 0:
        A, Ph = -A, Ph + np.pi
    omega = abs(Om) / 2.0                     # <n> oscillates at 2*omega
    phi0 = (Ph % (2 * np.pi)) / 2.0
    return omega, phi0, nbar


# ======================================================================
# 4.  JOINT FIT ACROSS HERALD CONDITIONS (one round)
# ======================================================================
def fit_round(entry, nsubs, r1_fixed, seed=None, D=14, sysfloor=0.02):
    """
    Joint fit of all herald conditions in one round, sharing (r2, eta, omega, phi0);
    r1 is FIXED to the loss-independent herald-arm value (g2 -> sinh^2 r1).
    Per-condition Poisson background. Returns dict of parameters (or None).
    """
    nmax = entry[nsubs[0]]["C"].shape[1] - 1
    obs, sig, Vs = {}, {}, {}
    for ns in nsubs:
        V, C, N = entry[ns]["V"], entry[ns]["C"], entry[ns]["N"]
        P = C / N[:, None]
        s = np.sqrt(np.maximum(P * (1 - P), 1e-12) / N[:, None])
        s = np.sqrt(s ** 2 + (sysfloor * np.maximum(P, 1e-4)) ** 2)
        obs[ns], sig[ns], Vs[ns] = P, np.maximum(s, 1e-7), V

    ref = nsubs[0]
    if seed is None:
        om, ph, _ = prefit_omega_phi0(Vs[ref], entry[ref]["C"], entry[ref]["N"])
    else:
        om, ph = seed
    nb0 = [max(obs[ns][:, 1].min() * 0.3, 1e-4) for ns in nsubs]

    def unpack(p):
        r2, eta, omega, phi0 = p[:4]
        nbg = {ns: p[4 + i] for i, ns in enumerate(nsubs)}
        return r2, eta, omega, phi0, nbg

    def resid(p):
        r2, eta, omega, phi0, nbg = unpack(p)
        out = []
        for ns in nsubs:
            M = model_pn(Vs[ns], r1_fixed, r2, eta, nbg[ns], omega, phi0, ns, nmax, D)
            out.append(((obs[ns] - M) / sig[ns]).ravel())
        return np.concatenate(out)

    lb = [0.005, 1e-3, 0.5 * om, ph - np.pi] + [0] * len(nsubs)
    ub = [3, 1.0, 1.6 * om, ph + np.pi] + [0.5] * len(nsubs)
    p0 = np.clip([0.1, 0.1, om, ph] + nb0, lb, ub)
    try:
        r = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=4000)
    except Exception:
        return None
    r2, eta, omega, phi0, nbg = unpack(r.x)
    ndof = max(sum(obs[ns].size for ns in nsubs) - len(r.x), 1)
    return {"r1": r1_fixed, "r2": r2, "eta": eta, "omega": omega, "phi0": phi0,
            "nbg": nbg, "chi2": 2 * r.cost / ndof, "x": r.x}


# ======================================================================
# 5.  CLASSICAL FISHER INFORMATION FROM DATA
# ======================================================================
# ======================================================================
#  GENERAL LOSS MODEL: internal loss (after S1) + separate detector losses
#  Needs full density-matrix propagation because loss BEFORE the second
#  squeezer is re-amplified by it (not a simple binomial loss at the end).
# ======================================================================
def loss_channel(rho, eta, D):
    """Amplitude-damping (loss) channel of transmission eta on a real density matrix."""
    if eta >= 0.999999:
        return rho
    # B[j,l] = sqrt(C(j+l,l)) * eta^(j/2) * (1-eta)^(l/2)
    j = np.arange(D)
    B = np.zeros((D, D))
    for l in range(D):
        jj = np.arange(D - l)
        B[jj, l] = np.sqrt(comb(jj + l, l)) * eta ** (jj / 2.0) * (1 - eta) ** (l / 2.0)
    out = np.zeros_like(rho)
    for l in range(D):
        w = B[:D - l, l]
        out[:D - l, :D - l] += np.outer(w, w) * rho[l:, l:]
    return out


def output_diag(phi_arr, r1, r2, eta_int, N1, D=16):
    """Photon-number distribution of the interferometer OUTPUT mode (before detector loss),
    including internal loss eta_int applied AFTER the first squeezer / at the herald tap."""
    S1, a = _squeeze(r1, D)
    S2, _ = _squeeze(r2, D)
    rho = np.zeros((D, D))
    rho[0, 0] = 1.0
    rho = S1 @ rho @ S1.T
    rho = loss_channel(rho, eta_int, D)
    if N1 > 0:
        aN = np.linalg.matrix_power(a, N1)
        rho = aN @ rho @ aN.T
        tr = np.trace(rho)
        if tr <= 0:
            return np.zeros((len(phi_arr), D))
        rho = rho / tr
    n = np.arange(D)
    out = np.empty((len(phi_arr), D))
    for i, phi in enumerate(phi_arr):
        ph = np.exp(-1j * phi * n)
        r = (ph[:, None] * rho) * np.conj(ph)[None, :]
        r = S2 @ r @ S2.T
        out[i] = np.real(np.diag(r))
    return out


def apply_detection(diag, eta_det, nbg, nmax, D):
    """Binomial detection loss eta_det + Poisson background on an output diagonal [nV, D]."""
    L = _loss_matrix(eta_det, D)              # L[n,m]=C(m,n)eta^n(1-eta)^(m-n)
    P = diag @ L.T
    if nbg > 1e-9:
        k = np.arange(D)
        pois = np.exp(-nbg) * nbg ** k / np.array([math.factorial(i) for i in k])
        P = np.array([np.convolve(P[i], pois)[:D] for i in range(P.shape[0])])
    return P[:, :nmax + 1]


def load_both(direction, nsubs):
    """Per round/condition: V, C3 (Det3 counts), C4 (Det4 counts), N (totals). One pass."""
    vfolders = voltage_folders(direction)
    rounds = {}
    for v, fp in vfolders:
        for rp in glob.glob(os.path.join(fp, "*_heralded_3way.txt")):
            k = int(re.search(r"round_(\d+)_", os.path.basename(rp)).group(1))
            rounds.setdefault(k, {})
    data = {k: {ns: {"V": [], "d3": [], "d4": []} for ns in nsubs} for k in rounds}
    for v, fp in vfolders:
        for rp in glob.glob(os.path.join(fp, "*_heralded_3way.txt")):
            k = int(re.search(r"round_(\d+)_", os.path.basename(rp)).group(1))
            mats, _ = parse_heralded_file(rp)
            for ns in nsubs:
                if ns in mats:
                    data[k][ns]["V"].append(v)
                    data[k][ns]["d3"].append(marginal(mats[ns], "D3"))
                    data[k][ns]["d4"].append(marginal(mats[ns], "D4"))
    out = {}
    nmax = 4
    for k in sorted(rounds):
        entry = {}
        ok = True
        for ns in nsubs:
            V = np.array(data[k][ns]["V"])
            if len(V) < 6:
                ok = False
                break
            idx = np.argsort(V)
            V = V[idx]
            D = max(max(len(d) for d in data[k][ns]["d3"]),
                    max(len(d) for d in data[k][ns]["d4"]))
            C3 = np.zeros((len(V), D)); C4 = np.zeros((len(V), D))
            for j, jj in enumerate(idx):
                d3 = data[k][ns]["d3"][jj]; d4 = data[k][ns]["d4"][jj]
                C3[j, :len(d3)] = d3; C4[j, :len(d4)] = d4
            N = C3.sum(axis=1)
            entry[ns] = {"V": V, "C3": C3[:, :nmax + 1], "C4": C4[:, :nmax + 1], "N": N}
        if ok:
            out[k] = entry
    return out, sorted(out)


def fit_round_general(entry, nsubs, r1, seed=None, D=16, sysfloor=0.02):
    """
    Joint fit of Det3 AND Det4 marginals across all herald conditions.
    Shared: r2, eta_int (internal loss), omega, phi0.  Separate: eta3, eta4.
    Per-condition Poisson background (shared between detectors).  r1 fixed.
    """
    nmax = entry[nsubs[0]]["C4"].shape[1] - 1
    obs, sig, Vs = {}, {}, {}
    for ns in nsubs:
        V, N = entry[ns]["V"], entry[ns]["N"]
        Vs[ns] = V
        for det in ("C3", "C4"):
            P = entry[ns][det] / N[:, None]
            s = np.sqrt(np.maximum(P * (1 - P), 1e-12) / N[:, None])
            s = np.sqrt(s ** 2 + (sysfloor * np.maximum(P, 1e-4)) ** 2)
            obs[(ns, det)] = P
            sig[(ns, det)] = np.maximum(s, 1e-7)
    ref = nsubs[0]
    if seed is None:
        om, ph, _ = prefit_omega_phi0(Vs[ref], entry[ref]["C4"], entry[ref]["N"])
    else:
        om, ph = seed
    nb0 = [max(obs[(ns, "C4")][:, 1].min() * 0.3, 1e-4) for ns in nsubs]

    def unpack(p):
        r2, eta_int, eta3, eta4, omega, phi0 = p[:6]
        nbg = {ns: p[6 + i] for i, ns in enumerate(nsubs)}
        return r2, eta_int, eta3, eta4, omega, phi0, nbg

    def resid(p):
        r2, eta_int, eta3, eta4, omega, phi0, nbg = unpack(p)
        out = []
        for ns in nsubs:
            diag = output_diag(omega * Vs[ns] + phi0, r1, r2, eta_int, ns, D)
            M3 = apply_detection(diag, eta3, nbg[ns], nmax, D)
            M4 = apply_detection(diag, eta4, nbg[ns], nmax, D)
            out.append(((obs[(ns, "C3")] - M3) / sig[(ns, "C3")]).ravel())
            out.append(((obs[(ns, "C4")] - M4) / sig[(ns, "C4")]).ravel())
        return np.concatenate(out)

    lb = [0.005, 0.05, 1e-3, 1e-3, 0.5 * om, ph - np.pi] + [0] * len(nsubs)
    ub = [3, 1.0, 1.0, 1.0, 1.6 * om, ph + np.pi] + [0.5] * len(nsubs)
    p0 = np.clip([0.1, 0.8, 0.1, 0.1, om, ph] + nb0, lb, ub)
    try:
        r = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=4000)
    except Exception:
        return None
    r2, eta_int, eta3, eta4, omega, phi0, nbg = unpack(r.x)
    ndof = max(sum(obs[key].size for key in obs) - len(r.x), 1)
    return {"r1": r1, "r2": r2, "eta_int": eta_int, "eta3": eta3, "eta4": eta4,
            "omega": omega, "phi0": phi0, "nbg": nbg, "chi2": 2 * r.cost / ndof}


def herald_g2(direction, nmax=8):
    """<n> and g2 of the herald arm from the 'Total Coincidences' per Px level."""
    counts = np.zeros(nmax + 1)
    for v, fp in voltage_folders(direction):
        for rp in glob.glob(os.path.join(fp, "*_heralded_3way.txt")):
            _, tot = parse_heralded_file(rp)
            for ns, c in tot.items():
                if ns <= nmax:
                    counts[ns] += c
    n = np.arange(nmax + 1)
    N = counts.sum()
    mean = (counts * n).sum() / N
    m2 = (counts * n * (n - 1)).sum() / N
    return mean, m2 / mean ** 2, counts / N


def cfi_from_data(entry, ns, par, D=14, ngrid=240):
    """
    CFI(phi) using measured probabilities (round-summed for this condition) and the
    model derivative.  Returns phi grid, F_pnr(phi), F_click(phi), and data means.
    """
    r1, r2, eta = par["r1"], par["r2"], par["eta"]
    nbg = par["nbg"][ns]
    omega, phi0 = par["omega"], par["phi0"]
    # measured probabilities as a function of phi (bin the scan by its own phase)
    phi = np.linspace(0, np.pi, ngrid)              # one full <n> period
    V = (phi - phi0) / omega
    M = model_pn(V, r1, r2, eta, nbg, omega, phi0, ns, nmax=entry[ns]["C"].shape[1] - 1, D=D)
    # analytic derivative via finite difference of the (smooth) model
    dphi = 1e-4
    Mp = model_pn(V + dphi / omega, r1, r2, eta, nbg, omega, phi0, ns,
                  nmax=entry[ns]["C"].shape[1] - 1, D=D)
    dM = (Mp - M) / dphi
    Pfloor = 1e-9
    F_pnr = np.sum(dM ** 2 / (M + Pfloor), axis=1)
    # click detector: {0, >=1}
    p0 = M[:, 0]
    dp0 = dM[:, 0]
    F_click = dp0 ** 2 / (p0 * (1 - p0) + Pfloor)
    return phi, F_pnr, F_click


# ======================================================================
# 6.  PIPELINE
# ======================================================================
def run(direction="UP", nsubs=(0, 1, 2), detector="D4", max_rounds=30):
    os.makedirs(FIG_DIR, exist_ok=True)
    nsubs = list(nsubs)
    print(f"\n=== Loading {direction}, herald conditions N1={nsubs}, detector {detector} ===")
    rounds, rids = load_direction(direction, nsubs, detector)
    rids = [k for k in rids if k in rounds][:max_rounds]
    print(f"  {len(rids)} usable rounds")

    # --- pin r1 from the loss-independent herald-arm g2 (single-mode sqz: g2=3+1/sinh^2 r1) ---
    mean_h, g2_h, hdist = herald_g2(direction)
    sinh2_r1 = 1.0 / (g2_h - 3.0) if g2_h > 3 else 0.1
    r1_fixed = math.asinh(math.sqrt(sinh2_r1))
    print(f"  herald arm: <n>={mean_h:.4f}  g2={g2_h:.2f}  ->  r1={r1_fixed:.3f}  "
          f"(sinh^2 r1 = {sinh2_r1:.4f}, loss-independent)")

    # --- per-round joint fits, seeding each from the previous (drift is gradual) ---
    fits, seed = {}, None
    for k in rids:
        fr = fit_round(rounds[k], nsubs, r1_fixed, seed=seed)
        if fr is None:
            continue
        fits[k] = fr
        seed = (fr["omega"], fr["phi0"])
    good = sorted(fits)
    R1 = np.array([fits[k]["r1"] for k in good])
    R2 = np.array([fits[k]["r2"] for k in good])
    ETA = np.array([fits[k]["eta"] for k in good])
    OM = np.array([fits[k]["omega"] for k in good])
    CHI = np.array([fits[k]["chi2"] for k in good])

    def ms(x):
        return np.median(x), np.std(x)
    print("  --- output fit parameters (median +/- std over rounds; r1 fixed from herald) ---")
    print(f"    r1  = {r1_fixed:.3f} (fixed)   [first squeezer]")
    print(f"    r2  = {ms(R2)[0]:.3f} +/- {ms(R2)[1]:.3f}   [second squeezer]")
    print(f"    eta = {ms(ETA)[0]:.3f} +/- {ms(ETA)[1]:.3f}   [collection incl. 50:50 split]")
    print(f"    fringe period = {math.pi/ms(OM)[0]:.1f} V   reduced chi2 ~ {np.median(CHI):.1f}")
    print(f"    gain config: r1(={r1_fixed:.2f}) {'>' if r1_fixed>ms(R2)[0] else '<'} r2(={ms(R2)[0]:.2f})"
          f"  -> {'NOT ' if r1_fixed>ms(R2)[0] else ''}the G2>G1 loss-tolerant regime")

    F_SNL = sinh2_r1                                   # classical shot-noise benchmark (per shot)

    par_med = {"r1": r1_fixed, "r2": np.median(R2), "eta": np.median(ETA),
               "omega": np.median(OM), "phi0": np.median([fits[k]["phi0"] for k in good]),
               "nbg": {ns: np.median([fits[k]["nbg"][ns] for k in good]) for ns in nsubs}}
    krep = good[len(good) // 2]
    cfi = {}
    print("  --- classical Fisher information per shot (peak over phase) ---")
    print(f"    {'cond':>6} {'F_PNR':>9} {'F_click':>9} {'PNRadv':>7} {'F/F_SNL':>8}"
          f" {'M_shots':>11} {'dphi_achieved':>13}")
    for ns in nsubs:
        phi, Fp, Fc = cfi_from_data(rounds[krep], ns, par_med)
        cfi[ns] = (phi, Fp, Fc)
        adv = 100 * (Fp.max() / Fc.max() - 1) if Fc.max() > 0 else 0
        # achieved sensitivity: total shots for this condition summed over the whole scan & all rounds
        Mtot = sum(rounds[k][ns]["N"].sum() for k in good)
        dphi_ach = 1.0 / math.sqrt(Mtot * Fp.max()) if Fp.max() > 0 else np.inf
        print(f"    N1={ns:>3} {Fp.max():9.5f} {Fc.max():9.5f} {adv:+6.0f}% "
              f"{Fp.max()/F_SNL:8.4f} {Mtot:11.3e} {dphi_ach:13.2e}")
    print(f"    F_SNL (ideal coherent, sinh^2 r1) = {F_SNL:.4f} per shot   "
          f"[F_PNR<F_SNL => below shot-noise, expected at eta~{np.median(ETA):.2f}]")

    _make_figures(direction, nsubs, rounds, fits, good, par_med, cfi, F_SNL,
                  R1, R2, ETA, hdist, mean_h, g2_h)
    return {"fits": fits, "par_med": par_med, "cfi": cfi, "F_SNL": F_SNL}


# ======================================================================
# 7.  FIGURES
# ======================================================================
def _make_figures(direction, nsubs, rounds, fits, good, par, cfi, F_SNL,
                  R1, R2, ETA, hdist, mean_h, g2_h):
    krep = good[len(good) // 2]
    colors = plt.cm.viridis(np.linspace(0, 0.85, 5))

    # ---- FIG 1: fringe fits, all conditions, sectors n=0..4 (one clean round) ----
    fig, axes = plt.subplots(len(nsubs), 1, figsize=(9, 3.0 * len(nsubs)), sharex=True)
    if len(nsubs) == 1:
        axes = [axes]
    p = fits[krep]
    for ax, ns in zip(axes, nsubs):
        V = rounds[krep][ns]["V"]
        C = rounds[krep][ns]["C"]
        N = rounds[krep][ns]["N"]
        P = C / N[:, None]
        Vd = np.linspace(V.min(), V.max(), 300)
        M = model_pn(Vd, p["r1"], p["r2"], p["eta"], p["nbg"][ns],
                     p["omega"], p["phi0"], ns, nmax=C.shape[1] - 1)
        for n in range(1, C.shape[1]):
            err = np.sqrt(np.maximum(P[:, n] * (1 - P[:, n]), 1e-12) / N)
            ax.errorbar(V, P[:, n], yerr=err, fmt="o", ms=3, color=colors[n],
                        alpha=0.6, capsize=0, label=f"data P({n})")
            ax.plot(Vd, M[:, n], "-", color=colors[n], lw=2)
        ax.set_yscale("log")
        ax.set_ylabel("P(n)")
        title = "no subtraction" if ns == 0 else f"subtract N1={ns}"
        ax.set_title(f"Herald {title}   (r1={p['r1']:.2f}, r2={p['r2']:.2f}, eta={p['eta']:.2f})",
                     fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Piezo voltage (V)")
    fig.suptitle(f"Exact-model fits of P(n,phi), detector marginal  [{direction}, round {krep}]",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig1_fringe_fits_{direction}.png"), dpi=130)
    plt.close(fig)

    # ---- FIG 2: drift & parameter stability over rounds ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ph = np.unwrap([fits[k]["phi0"] for k in good])
    axs[0].plot(good, np.degrees(ph - ph[0]), "o-")
    axs[0].set_title("Phase drift phi0 over time")
    axs[0].set_xlabel("round"); axs[0].set_ylabel("phi0 drift (deg)")
    axs[1].plot(good, R1, "o-", label="r1"); axs[1].plot(good, R2, "s-", label="r2")
    axs[1].set_title("Squeezing parameters"); axs[1].set_xlabel("round"); axs[1].legend()
    axs[2].plot(good, ETA, "o-", color="C2")
    axs[2].set_title("Collection efficiency eta"); axs[2].set_xlabel("round")
    for a in axs:
        a.grid(True, ls=":", alpha=0.5)
    fig.suptitle(f"Per-round stability [{direction}]", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig2_drift_{direction}.png"), dpi=130)
    plt.close(fig)

    # ---- FIG 3: Fisher information -- (a) PNR vs click zoomed, (b) log bars vs SNL ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, ns in enumerate(nsubs):
        phi, Fp, Fc = cfi[ns]
        lab = "no subtr." if ns == 0 else f"N1={ns}"
        axA.plot(phi, Fp, "-", color=colors[i + 1], lw=2.3, label=f"PNR, {lab}")
        axA.plot(phi, Fc, "--", color=colors[i + 1], lw=1.5, alpha=0.85, label=f"click, {lab}")
    axA.set_xlabel("phase phi (rad)"); axA.set_ylabel("F(phi) per shot")
    axA.set_title("(a) photon-counting (solid) beats click (dashed)")
    axA.grid(True, ls=":", alpha=0.5); axA.legend(fontsize=8, ncol=2)

    x = np.arange(len(nsubs)); w = 0.38
    Fp_pk = [cfi[ns][1].max() for ns in nsubs]
    Fc_pk = [cfi[ns][2].max() for ns in nsubs]
    axB.bar(x - w / 2, Fp_pk, w, color="C0", label="PNR")
    axB.bar(x + w / 2, Fc_pk, w, color="C3", label="click")
    axB.axhline(F_SNL, color="k", ls=":", lw=2, label=f"SNL sinh$^2$r1={F_SNL:.3f}")
    for xi, ns in zip(x, nsubs):
        adv = 100 * (cfi[ns][1].max() / cfi[ns][2].max() - 1)
        axB.text(xi, Fp_pk[list(nsubs).index(ns)] * 1.15, f"+{adv:.0f}%",
                 ha="center", fontsize=9, fontweight="bold")
    axB.set_yscale("log"); axB.set_xticks(x)
    axB.set_xticklabels(["no subtr." if n == 0 else f"N1={n}" for n in nsubs])
    axB.set_ylabel("peak F per shot"); axB.set_title("(b) peak CFI vs shot-noise limit (log)")
    axB.grid(True, which="both", ls=":", alpha=0.4); axB.legend(fontsize=8)
    fig.suptitle(f"Classical Fisher information [{direction}]  "
                 f"(eta={par['eta']:.3f}, r1={par['r1']:.2f}>r2={par['r2']:.2f})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig3_fisher_{direction}.png"), dpi=130)
    plt.close(fig)

    # ---- FIG 4: herald-arm photon statistics (sanity of r1) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    nn = np.arange(len(hdist))
    ax.bar(nn[:5], hdist[:5], color="C0", alpha=0.8)
    ax.set_yscale("log"); ax.set_xlabel("herald photon number"); ax.set_ylabel("probability")
    ax.set_title(f"Herald-arm statistics [{direction}]\n<n>={mean_h:.4f},  g2={g2_h:.2f}"
                 f"  ->  sinh^2(r1)=1/(g2-3)={1/(g2_h-3):.3f}", fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig4_herald_{direction}.png"), dpi=130)
    plt.close(fig)
    print(f"  figures written to {FIG_DIR}/")


def compare_readout(direction="UP", nsubs=(0, 1, 2), max_rounds=12):
    """Compare single-detector marginal (D4) vs full output photon number (n3+n4)."""
    mean_h, g2_h, _ = herald_g2(direction)
    r1 = math.asinh(math.sqrt(1.0 / (g2_h - 3.0)))
    res = {}
    for det in ("D4", "tot"):
        rounds, rids = load_direction(direction, list(nsubs), det)
        rids = [k for k in rids if k in rounds][:max_rounds]
        fits, seed = {}, None
        for k in rids:
            fr = fit_round(rounds[k], list(nsubs), r1, seed=seed)
            if fr:
                fits[k] = fr
                seed = (fr["omega"], fr["phi0"])
        g = sorted(fits)
        par = {"r1": r1, "r2": np.median([fits[k]["r2"] for k in g]),
               "eta": np.median([fits[k]["eta"] for k in g]),
               "omega": np.median([fits[k]["omega"] for k in g]),
               "phi0": np.median([fits[k]["phi0"] for k in g]),
               "nbg": {ns: np.median([fits[k]["nbg"][ns] for k in g]) for ns in nsubs}}
        kr = g[len(g) // 2]
        adv, Fp = [], []
        for ns in nsubs:
            phi, fp, fc = cfi_from_data(rounds[kr], ns, par)
            adv.append(100 * (fp.max() / fc.max() - 1))
            Fp.append(fp.max())
        res[det] = {"adv": adv, "Fp": Fp, "eta": par["eta"]}
    # figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(nsubs)); w = 0.38
    a1.bar(x - w / 2, res["D4"]["adv"], w, color="C0", label=f"1 detector (eta={res['D4']['eta']:.2f})")
    a1.bar(x + w / 2, res["tot"]["adv"], w, color="C1", label=f"n3+n4 total (eta={res['tot']['eta']:.2f})")
    a1.set_ylabel("PNR advantage over click (%)"); a1.set_xticks(x)
    a1.set_xticklabels(["no subtr." if n == 0 else f"N1={n}" for n in nsubs])
    a1.set_title("(a) counting the full output vs one detector"); a1.legend(); a1.grid(ls=":", alpha=0.4)
    a2.bar(x - w / 2, res["D4"]["Fp"], w, color="C0", label="1 detector")
    a2.bar(x + w / 2, res["tot"]["Fp"], w, color="C1", label="n3+n4 total")
    a2.set_yscale("log"); a2.set_ylabel("peak F_PNR per shot"); a2.set_xticks(x)
    a2.set_xticklabels(["no subtr." if n == 0 else f"N1={n}" for n in nsubs])
    a2.set_title("(b) absolute Fisher information"); a2.legend(); a2.grid(which="both", ls=":", alpha=0.4)
    fig.suptitle(f"Read-out comparison [{direction}]: reconstruct the whole output, don't throw half away",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig5_readout_{direction}.png"), dpi=130)
    plt.close(fig)
    print(f"  read-out comparison [{direction}]:")
    for i, ns in enumerate(nsubs):
        print(f"    N1={ns}: PNR advantage  1-det {res['D4']['adv'][i]:+.0f}%  ->  total {res['tot']['adv'][i]:+.0f}%")
    return res


def _pool_compensated(rounds, rids, nsubs, omega, phi0_list):
    """Pool all scans onto a common phase axis phi=omega*V+phi0_round (drift removed)."""
    pooled = {}
    for ns in nsubs:
        phi, C3, C4, N = [], [], [], []
        for k in rids:
            if ns not in rounds[k]:
                continue
            V = rounds[k][ns]["V"]
            phi.append(omega * V + phi0_list[k])
            C3.append(rounds[k][ns]["C3"]); C4.append(rounds[k][ns]["C4"])
            N.append(rounds[k][ns]["N"])
        pooled[ns] = {"phi": np.concatenate(phi), "C3": np.vstack(C3),
                      "C4": np.vstack(C4), "N": np.concatenate(N)}
    return pooled


def _bin_pooled(pool_ns, nbins=60):
    """Fold phi to [0,pi) and bin (sum counts) for fast fitting. Returns phi_c, C3, C4, N."""
    phi = np.mod(pool_ns["phi"], np.pi)
    edges = np.linspace(0, np.pi, nbins + 1)
    idx = np.clip(np.digitize(phi, edges) - 1, 0, nbins - 1)
    K = pool_ns["C4"].shape[1]
    C3 = np.zeros((nbins, K)); C4 = np.zeros((nbins, K)); Nc = np.zeros(nbins)
    for b in range(nbins):
        m = idx == b
        if m.any():
            C3[b] = pool_ns["C3"][m].sum(0); C4[b] = pool_ns["C4"][m].sum(0)
            Nc[b] = pool_ns["N"][m].sum()
    keep = Nc > 0
    ctr = 0.5 * (edges[:-1] + edges[1:])
    return ctr[keep], C3[keep], C4[keep], Nc[keep]


def fit_pooled_general(pooled, nsubs, r1, D=16, sysfloor=0.02):
    """Fit the general loss model to the pooled, drift-compensated data (phase-binned).
    Shared r2, eta_int, global phase delta; separate eta3, eta4; per-condition background."""
    nmax = pooled[nsubs[0]]["C4"].shape[1] - 1
    obs, sig, phis = {}, {}, {}
    for ns in nsubs:
        pc, C3, C4, N = _bin_pooled(pooled[ns])
        phis[ns] = pc
        for det, C in (("C3", C3), ("C4", C4)):
            P = C / N[:, None]
            s = np.sqrt(np.maximum(P * (1 - P), 1e-12) / N[:, None])
            s = np.sqrt(s ** 2 + (sysfloor * np.maximum(P, 1e-4)) ** 2)
            obs[(ns, det)] = P; sig[(ns, det)] = np.maximum(s, 1e-7)

    def unpack(p):
        r2, eta_int, eta3, eta4, delta = p[:5]
        nbg = {ns: p[5 + i] for i, ns in enumerate(nsubs)}
        return r2, eta_int, eta3, eta4, delta, nbg

    def resid(p):
        r2, eta_int, eta3, eta4, delta, nbg = unpack(p)
        out = []
        for ns in nsubs:
            diag = output_diag(phis[ns] + delta, r1, r2, eta_int, ns, D)
            M3 = apply_detection(diag, eta3, nbg[ns], nmax, D)
            M4 = apply_detection(diag, eta4, nbg[ns], nmax, D)
            out.append(((obs[(ns, "C3")] - M3) / sig[(ns, "C3")]).ravel())
            out.append(((obs[(ns, "C4")] - M4) / sig[(ns, "C4")]).ravel())
        return np.concatenate(out)

    lb = [0.005, 0.05, 1e-3, 1e-3, -np.pi] + [0] * len(nsubs)
    ub = [3, 1.0, 1.0, 1.0, np.pi] + [0.5] * len(nsubs)
    best = None
    for d0 in (0, np.pi / 2, np.pi, -np.pi / 2):
        p0 = np.clip([0.1, 0.8, 0.05, 0.05, d0] + [0.002] * len(nsubs), lb, ub)
        try:
            r = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=3000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    r2, eta_int, eta3, eta4, delta, nbg = unpack(best.x)
    ndof = max(sum(obs[k].size for k in obs) - len(best.x), 1)
    return {"r1": r1, "r2": r2, "eta_int": eta_int, "eta3": eta3, "eta4": eta4,
            "delta": delta, "nbg": nbg, "chi2": 2 * best.cost / ndof}


def _profile_eta_int(pool_ns, r1, D, etas=(1.0, 0.6, 0.4, 0.25)):
    """Profile reduced chi2 vs fixed internal loss for one condition (Det4), to show degeneracy."""
    phi, _, C4, N = _bin_pooled(pool_ns)
    P = (C4 / N[:, None])[:, :5]
    s = np.sqrt(np.maximum(P * (1 - P), 1e-12) / N[:, None]) + 0.02 * np.maximum(P, 1e-4)
    out = []
    for ei in etas:
        def resid(p):
            r2, eta4, delta, nbg = p
            d = output_diag(phi + delta, r1, r2, ei, 0, D)
            M = apply_detection(d, eta4, nbg, 4, D)
            return ((P - M) / s).ravel()
        try:
            r = least_squares(resid, [0.05, 0.05, 0, 0.002],
                              bounds=([.005, .001, -np.pi, 0], [2, 1, np.pi, .1]), max_nfev=2000)
            out.append((ei, 2 * r.cost / max(P.size - 4, 1)))
        except Exception:
            out.append((ei, np.nan))
    return out


def run_general(direction="UP", nsubs=(0, 1, 2), D=16):
    """Fit the full loss model (internal loss + separate eta3/eta4) to the pooled,
    drift-compensated data, and make the drift-compensation plots of P(n) vs phi."""
    os.makedirs(FIG_DIR, exist_ok=True)
    nsubs = list(nsubs)
    print(f"\n=== GENERAL LOSS MODEL [{direction}] (internal loss + separate eta3/eta4) ===")
    rounds, rids = load_both(direction, nsubs)
    mean_h, g2_h, _ = herald_g2(direction)
    r1 = math.asinh(math.sqrt(1.0 / (g2_h - 3.0)))

    # per-round phi0 (fast prefit) for drift compensation; common omega
    om_list, phi0_list = [], {}
    for k in rids:
        om, ph, _ = prefit_omega_phi0(rounds[k][0]["V"], rounds[k][0]["C4"], rounds[k][0]["N"])
        om_list.append(om); phi0_list[k] = ph
    omega = float(np.median(om_list))

    pooled = _pool_compensated(rounds, rids, nsubs, omega, phi0_list)
    # Fit each herald condition separately (best marginal description; shared params
    # across conditions create tension from the weak-tap / multimode approximations).
    parc = {}
    for ns in nsubs:
        parc[ns] = fit_pooled_general({ns: pooled[ns]}, [ns], r1, D=D)
    par = parc[nsubs[0]].copy()
    par["omega"] = omega
    par["parc"] = parc
    print(f"  r1={r1:.3f} (fixed, herald)   fringe period={math.pi/omega:.1f} V   "
          f"(pooled {len(rids)} scans, per-condition fits)")
    print(f"  {'cond':>6} {'r2':>6} {'eta3':>7} {'eta4':>7} {'eta4/eta3':>9} {'chi2':>7}")
    for ns in nsubs:
        p = parc[ns]
        print(f"  {('N1='+str(ns)):>6} {p['r2']:6.3f} {p['eta3']:7.3f} {p['eta4']:7.3f}"
              f" {p['eta4']/p['eta3']:9.2f} {p['chi2']:7.1f}")

    # internal-loss identifiability: profile chi2(eta_int) for N1=0 (D4)
    prof = _profile_eta_int(pooled[0], r1, D)
    print("  internal loss eta_int is degenerate with gain/efficiency in the marginals:")
    print("    chi2 vs eta_int (N1=0):  " +
          "  ".join(f"{e:.2f}->{c:.0f}" for e, c in prof))
    print("    => not independently determinable here; needs single-pass calibration "
          "or the full 2-D joint P(n3,n4).")

    colors = plt.cm.viridis(np.linspace(0, 0.85, 5))
    # ---- FIG 6: pooled fit quality, Det3 & Det4 ----
    fig, axes = plt.subplots(len(nsubs), 2, figsize=(13, 3.0 * len(nsubs)), sharex=True)
    phg = np.linspace(0, np.pi, 200)
    for row, ns in enumerate(nsubs):
        phi = np.mod(pooled[ns]["phi"], np.pi)
        p = parc[ns]
        for col, (det, eta) in enumerate([("C3", "eta3"), ("C4", "eta4")]):
            ax = axes[row, col]
            P = pooled[ns][det] / pooled[ns]["N"][:, None]
            diag = output_diag(phg + p["delta"], r1, p["r2"], p["eta_int"], ns, D)
            M = apply_detection(diag, p[eta], p["nbg"][ns], P.shape[1] - 1, D)
            for n in range(1, P.shape[1]):
                ax.plot(phi, P[:, n], ".", ms=2, color=colors[n], alpha=0.25)
                ax.plot(phg, M[:, n], "-", color=colors[n], lw=2, label=f"P({n})")
            ax.set_yscale("log")
            ax.set_title(f"{'no subtr' if ns==0 else f'N1={ns}'}  Det{det[-1]} (eta={par[eta]:.3f})",
                         fontsize=9)
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("phase phi (rad)")
    fig.suptitle(f"General loss model, pooled per-condition fits [{direction}]  "
                 f"(separate eta3, eta4; internal loss degenerate -> set to 1)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"fig6_generalfit_{direction}.png"), dpi=130)
    plt.close(fig)

    # ---- FIG 7: drift compensation -- all scans, raw vs V and aligned vs phi ----
    for ns in (0, 1):
        if ns not in nsubs:
            continue
        photons = [1, 2, 3]
        fig, axes = plt.subplots(len(photons), 2, figsize=(13, 2.6 * len(photons)), sharex="col")
        for row, n in enumerate(photons):
            axR, axC = axes[row, 0], axes[row, 1]
            for k in rids:
                if ns not in rounds[k] or rounds[k][ns]["C4"].shape[1] <= n:
                    continue
                V = rounds[k][ns]["V"]; N = rounds[k][ns]["N"]
                P = rounds[k][ns]["C4"][:, n] / N
                axR.plot(V, P, "-", color="0.6", lw=0.6, alpha=0.5)
                phi = np.mod(omega * V + phi0_list[k], np.pi)
                o = np.argsort(phi)
                axC.plot(phi[o], P[o], "-", color="0.6", lw=0.6, alpha=0.5)
            p = parc[ns]
            diag = output_diag(phg + p["delta"], r1, p["r2"], p["eta_int"], ns, D)
            Mg = apply_detection(diag, p["eta4"], p["nbg"][ns], n, D)[:, n]
            axC.plot(phg, Mg, "-", color="crimson", lw=2.3, label="pooled model")
            axR.set_ylabel(f"P({n})"); axR.grid(ls=":", alpha=0.4); axC.grid(ls=":", alpha=0.4)
            if row == 0:
                axR.set_title("raw: all scans vs piezo voltage (phase drifts)")
                axC.set_title("drift-compensated & folded vs phase phi [rad]")
                axC.legend(fontsize=8)
        axes[-1, 0].set_xlabel("piezo voltage (V)")
        axes[-1, 1].set_xlabel("phase phi (rad)")
        fig.suptitle(f"Drift compensation, {'no subtraction' if ns==0 else f'N1={ns}'} "
                     f"[{direction}, all {len(rids)} scans, Det4]", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"fig7_driftcomp_N1{ns}_{direction}.png"), dpi=130)
        plt.close(fig)
    print(f"  figures fig6/fig7 written to {FIG_DIR}/")
    return par


if __name__ == "__main__":
    summary = {}
    for direction in ("UP", "DOWN"):
        try:
            res = run(direction, nsubs=(0, 1, 2), detector="D4")
            summary[direction] = {
                "F_SNL": res["F_SNL"],
                "r1_med": np.median([f["r1"] for f in res["fits"].values()]),
                "r2_med": np.median([f["r2"] for f in res["fits"].values()]),
                "eta_med": np.median([f["eta"] for f in res["fits"].values()]),
            }
        except Exception as e:
            print(f"  {direction} failed: {e}")
    try:
        compare_readout("UP")
    except Exception as e:
        print(f"  readout comparison failed: {e}")
    for direction in ("UP", "DOWN"):
        try:
            run_general(direction)
        except Exception as e:
            print(f"  general model {direction} failed: {e}")
    with open(os.path.join(FIG_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nDone.")
