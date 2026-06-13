"""Monte Carlo robustness and hyperparameter sensitivity for the
weak vs strong SINDy Burgers benchmark (reproduces manuscript setup).

Benchmark: u_t = -u u_x + nu u_xx, nu=0.075, x in [0,2pi), Nx=128,
Nt=241, T=1.2, RK4 + Fourier spectral derivatives.
IC: 0.90 sin x + 0.35 sin(2x+0.3) - 0.18 cos(3x-0.1).
Noise: eta ~ N(0, (sigma*std(u))^2), sigma in {0.01,0.03,0.05,0.08}.
"""
import numpy as np
import json
from pathlib import Path

rng_master = np.random.default_rng(20260612)
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "legacy_mc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- reference simulation ----------------
Nx, Nt, T, nu = 128, 241, 1.20, 0.075
L = 2*np.pi
x = np.linspace(0, L, Nx, endpoint=False)
t = np.linspace(0, T, Nt)
dt = t[1]-t[0]
dx = x[1]-x[0]
k = np.fft.fftfreq(Nx, d=dx)*2*np.pi
ik, k2 = 1j*k, -(k**2)

def dxs(u):  return np.real(np.fft.ifft(ik*np.fft.fft(u)))
def dxxs(u): return np.real(np.fft.ifft(k2*np.fft.fft(u)))

def rhs(u):  return -u*dxs(u) + nu*dxxs(u)

u0 = 0.90*np.sin(x) + 0.35*np.sin(2*x+0.30) - 0.18*np.cos(3*x-0.10)
U = np.zeros((Nt, Nx)); U[0] = u0
for i in range(Nt-1):
    u = U[i]
    k1 = rhs(u); k2_ = rhs(u+0.5*dt*k1); k3 = rhs(u+0.5*dt*k2_); k4 = rhs(u+dt*k3)
    U[i+1] = u + dt/6*(k1+2*k2_+2*k3+k4)

std_u = U.std()
xi_star = np.array([0,0,0,0,-1.0,nu])   # [1,u,u^2,ux,uux,uxx]
TRUE_SUPP = frozenset({4,5})
TERMS = ["1","u","u2","ux","uux","uxx"]

# ---------------- STLSQ ----------------
def stlsq(A, b, lam, iters=12):
    norms = np.linalg.norm(A, axis=0); norms[norms==0]=1.0
    An = A/norms
    xi = np.linalg.lstsq(An, b, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi) < lam
        xi[small] = 0.0
        big = ~small
        if big.sum()==0: break
        xi[big] = np.linalg.lstsq(An[:,big], b, rcond=None)[0]
    return xi/norms

# ---------------- strong-form identification ----------------
def strong_identify(Ud, lam_s=0.1):
    ut = np.gradient(Ud, dt, axis=0)
    ux = np.array([dxs(r) for r in Ud])
    uxx = np.array([dxxs(r) for r in Ud])
    A = np.column_stack([np.ones(Ud.size), Ud.ravel(), (Ud**2).ravel(),
                         ux.ravel(), (Ud*ux).ravel(), uxx.ravel()])
    return stlsq(A, ut.ravel(), lam_s)

# ---------------- weak-form identification ----------------
def kernel(s, a):
    out = np.zeros_like(s)
    m = np.abs(s) <= a
    out[m] = 0.5*(1+np.cos(np.pi*s[m]/a))
    return out
def kernel_d1(s, a):
    out = np.zeros_like(s); m = np.abs(s) <= a
    out[m] = -0.5*np.pi/a*np.sin(np.pi*s[m]/a)
    return out
def kernel_d2(s, a):
    out = np.zeros_like(s); m = np.abs(s) <= a
    out[m] = -0.5*(np.pi/a)**2*np.cos(np.pi*s[m]/a)
    return out

def weak_identify(Ud, lam_w=0.1, ax=0.9, at=0.22, nxc=14, ntc=10):
    xc = np.linspace(ax, L-ax, nxc)
    tc = np.linspace(at, T-at, ntc)
    Xg, Tg = np.meshgrid(x, t)          # (Nt,Nx)
    rows_G, rows_b = [], []
    U2 = Ud**2
    for tm in tc:
        st = Tg - tm
        pt, pt_t = kernel(st, at), kernel_d1(st, at)
        for xm in xc:
            sx = Xg - xm
            px, px_x, px_xx = kernel(sx, ax), kernel_d1(sx, ax), kernel_d2(sx, ax)
            psi    = px*pt
            psi_t  = px*pt_t
            psi_x  = px_x*pt
            psi_xx = px_xx*pt
            w = dx*dt
            b  = -np.sum(Ud*psi_t)*w
            g1 =  np.sum(psi)*w
            g2 =  np.sum(Ud*psi)*w
            g3 =  np.sum(U2*psi)*w
            g4 = -np.sum(Ud*psi_x)*w          # ux feature (one IBP)
            g5 = -0.5*np.sum(U2*psi_x)*w      # u u_x = 1/2 (u^2)_x
            g6 =  np.sum(Ud*psi_xx)*w         # u_xx (two IBP)
            rows_G.append([g1,g2,g3,g4,g5,g6]); rows_b.append(b)
    return stlsq(np.array(rows_G), np.array(rows_b), lam_w)

# ---------------- rollout ----------------
def rollout_rmse(xi):
    c_adv, c_dif = xi[4], max(xi[5], 1e-6)
    # general library rollout with all terms
    def rhs_m(u):
        ux_, uxx_ = dxs(u), dxxs(u)
        return (xi[0] + xi[1]*u + xi[2]*u**2 + xi[3]*ux_ + xi[4]*u*ux_
                + max(xi[5],0.0)*uxx_)
    Um = np.zeros_like(U); Um[0]=u0
    try:
        for i in range(Nt-1):
            u = Um[i]
            k1=rhs_m(u);k2_=rhs_m(u+0.5*dt*k1);k3=rhs_m(u+0.5*dt*k2_);k4=rhs_m(u+dt*k3)
            Um[i+1]=u+dt/6*(k1+2*k2_+2*k3+k4)
            if not np.isfinite(Um[i+1]).all() or np.abs(Um[i+1]).max()>50:
                return np.nan
    except FloatingPointError:
        return np.nan
    return float(np.sqrt(np.mean((Um-U)**2)))

def Exi(xi): return float(np.linalg.norm(xi-xi_star)/np.linalg.norm(xi_star))
def supp_ok(xi): return frozenset(np.nonzero(np.abs(xi)>1e-10)[0]) == TRUE_SUPP

# ---------------- single-seed check vs paper ----------------
rng = np.random.default_rng(0)
for s in [0.0,0.05]:
    Ud = U + rng.normal(0, s*std_u, U.shape)
    xs_, xw_ = strong_identify(Ud), weak_identify(Ud)
    print(f"sigma={s}: strong uux={xs_[4]:.4f} uxx={xs_[5]:.4f} E={Exi(xs_):.3e} | "
          f"weak uux={xw_[4]:.4f} uxx={xw_[5]:.4f} E={Exi(xw_):.3e}")

# ---------------- Monte Carlo ----------------
NSEEDS = 30
sigmas = [0.01, 0.03, 0.05, 0.08]
results = {}
for s in sigmas:
    for meth in ["strong","weak"]:
        results[(s,meth)] = {"E":[], "R":[], "S":[]}
for seed in range(NSEEDS):
    rng = np.random.default_rng(1000+seed)
    for s in sigmas:
        Ud = U + rng.normal(0, s*std_u, U.shape)
        for meth, fn in [("strong",strong_identify),("weak",weak_identify)]:
            xi = fn(Ud)
            results[(s,meth)]["E"].append(Exi(xi))
            results[(s,meth)]["R"].append(rollout_rmse(xi))
            results[(s,meth)]["S"].append(supp_ok(xi))

summary = {}
for key,v in results.items():
    E = np.array(v["E"]); R = np.array([r for r in v["R"] if np.isfinite(r)])
    summary[f"{key[0]}_{key[1]}"] = dict(
        medE=float(np.median(E)), iqrE=float(np.subtract(*np.percentile(E,[75,25]))),
        q25E=float(np.percentile(E,25)), q75E=float(np.percentile(E,75)),
        medR=float(np.median(R)) if R.size else float("nan"),
        iqrR=float(np.subtract(*np.percentile(R,[75,25]))) if R.size else float("nan"),
        supp=float(np.mean(v["S"])))
    print(key, summary[f"{key[0]}_{key[1]}"])

np.save(OUTPUT_DIR / "mc_raw.npy", np.array([(s, m=="weak", e, r, sup)
        for (s,m),v in results.items() for e,r,sup in zip(v["E"],v["R"],v["S"])]))
with open(OUTPUT_DIR / "mc_summary.json","w") as f: json.dump(summary,f,indent=1)

# ---------------- hyperparameter sensitivity at sigma=0.05 ----------------
rngH = np.random.default_rng(777)
UdH = U + rngH.normal(0, 0.05*std_u, U.shape)
lams = np.array([0.01,0.02,0.05,0.1,0.2,0.4])
E_lam_w = [Exi(weak_identify(UdH, lam_w=l)) for l in lams]
E_lam_s = [Exi(strong_identify(UdH, lam_s=l)) for l in lams]
scales = np.array([0.5,0.75,1.0,1.25,1.5,2.0])
E_ax = [Exi(weak_identify(UdH, ax=0.9*c)) for c in scales]
E_at = [Exi(weak_identify(UdH, at=0.22*c)) for c in scales]
hyper = dict(lams=lams.tolist(), E_lam_w=E_lam_w, E_lam_s=E_lam_s,
             scales=scales.tolist(), E_ax=E_ax, E_at=E_at)
with open(OUTPUT_DIR / "hyper.json","w") as f: json.dump(hyper,f,indent=1)
print("hyper:", hyper)
