"""
Toy illustration of CTRL-style LLM-guided residual learning.

Pipeline (mirrors the paper at a cartoon level):
  1. A frozen backbone (ridge-regularised linear map, DLinear-like) makes base forecasts.
  2. A "controller" looks at how the backbone erred on HELD-OUT data (split into trend /
     seasonal parts) and emits control signals [scale, bias, gate, confidence].
  3. A tiny residual decoder turns those signals into a forecast correction.
  4. At test time, input-only statistics are z-scored against held-out statistics; if a
     component has shifted (> 2 sd) the controller is asked again and signals are adjusted.

NOTE: the controller here is a rule-based STAND-IN for the LLM (no API calls are made).
`controller_prompt` shows the text a real LLM agent would receive.
"""
import json
import os
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(int(os.environ.get("SEED", 7)))
QUIET = bool(os.environ.get("QUIET"))
P, L, H = 24, 96, 24          # season length, lookback, horizon

# ---------------------------------------------------------------- data
def make_series(n, slope, amp, noise=0.35, t0=0):
    t = np.arange(t0, t0 + n)
    return slope * t + amp * np.sin(2 * np.pi * t / P) + noise * rng.standard_normal(n)

n_train, n_val, n_test = 2400, 800, 800
# train + held-out: steady upward trend, seasonal amplitude 2.
# test: regime change -> seasonal amplitude jumps to 3.2 (trend unchanged).
SLOPE = 0.02
train = make_series(n_train, SLOPE, 2.0)
val   = make_series(n_val,   SLOPE, 2.0, t0=n_train)
level = SLOPE * (n_train + n_val)
t_test = np.arange(n_train + n_val, n_train + n_val + n_test)
test = (level + SLOPE * (t_test - t_test[0]) + 3.2 * np.sin(2 * np.pi * t_test / P)
        + 0.35 * rng.standard_normal(n_test))
full = np.concatenate([train, val, test])
bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, len(full))}

def windows(lo, hi, stride=H):
    X, Y = [], []
    for s in range(lo - lo % P, hi - L - H + 1, stride):
        X.append(full[s:s + L]); Y.append(full[s + L:s + L + H])
    return np.array(X), np.array(Y)

Xva, Yva = windows(*bounds["val"])
Xte, Yte = windows(*bounds["test"])

# ---------------------------------------------------------------- frozen backbone
# A stable, "fit once and freeze" model: it learned the average 24-step seasonal shape on the
# training data but (like many smooth models) shrinks it, and it damps the trend it extrapolates.
tt = np.arange(n_train)
detr = train - np.polyval(np.polyfit(tt, train, 1), tt)
profile = 0.8 * np.array([detr[k::P].mean() for k in range(P)])        # oversmoothed seasonal shape
def backbone(X):
    out = []
    for x in X:
        t0 = len(x)
        base = x[-P:].mean() - profile[(np.arange(t0 - P, t0) % P)].mean() * 0  # level of last cycle
        slope = 0.7 * np.polyfit(np.arange(len(x)), x, 1)[0]              # damped trend
        h = np.arange(H)
        out.append(base + slope * (h + P / 2) + profile[(t0 + h) % P])
    return np.array(out)

# ---------------------------------------------------------------- component split
_h = np.arange(H)
_B = np.stack([np.ones(H), _h, np.sin(2 * np.pi * _h / P), np.cos(2 * np.pi * _h / P)], 1)
def split(y):
    """Split a one-cycle (24-step) window into a trend line and a daily-cycle component by
    least squares on [1, t, sin, cos]. Returns (trend, seasonal, irregular-remainder)."""
    c = np.linalg.lstsq(_B, y, rcond=None)[0]
    tr = _B[:, :2] @ c[:2]
    se = _B[:, 2:] @ c[2:]
    return tr, se, y - tr - se

def input_stats(x):
    """Input-only statistics (no labels): trend slope and seasonal dominance from STL."""
    r = STL(x, period=P, robust=True).fit()
    slope = np.polyfit(np.arange(len(x)), r.trend, 1)[0]
    return slope, np.std(r.seasonal)          # (trend slope, seasonal strength)

# ---------------------------------------------------------------- "LLM" controller (stand-in)
def diagnose(Yhat, Y):
    """Runtime error statistics the LLM would be shown (never raw series or dataset name)."""
    def stats(Z):
        cs = np.array([np.linalg.lstsq(_B, z, rcond=None)[0] for z in Z])
        return cs[:, 1].mean(), np.hypot(cs[:, 2], cs[:, 3]).mean()
    sp, ap = stats(Yhat); st, at = stats(Y)
    return {"trend_slope_pred": sp, "trend_slope_true": st,
            "seas_amp_pred": ap, "seas_amp_true": at, "mae": float(np.mean(np.abs(Yhat - Y)))}

def controller_prompt(d):
    return (f"Backbone MAE was {d['mae']:.3f}. Predicted trend slope {d['trend_slope_pred']:.4f} vs "
            f"actual {d['trend_slope_true']:.4f}. Seasonal amplitude predicted {d['seas_amp_pred']:.2f} "
            f"vs actual {d['seas_amp_true']:.2f}. Output JSON [scale, bias, gate, confidence] per component.")

def controller(d):
    """Rule-based stand-in for the Trend / Seasonal agents."""
    clip = lambda v, a, b: float(np.clip(v, a, b))
    z = {}
    z["trend"] = dict(scale=clip(d["trend_slope_true"] / (d["trend_slope_pred"] + 1e-9), .5, 1.5),
                      bias=0.0, gate=0.6, conf=0.8)
    z["seasonal"] = dict(scale=clip(d["seas_amp_true"] / (d["seas_amp_pred"] + 1e-9), .5, 1.5),
                         bias=0.0, gate=0.6, conf=0.8)
    return z

def adapt_controller(z, shift):
    """Stand-in for the test-time adaptation prompt: shift scores in, adjusted signals out.
    Shifted components get a bigger scale move, more gate, and a confidence penalty --
    the same four knobs the paper's adaptation prompt exposes."""
    z = json.loads(json.dumps(z))
    for c, s in shift.items():
        if s > 2.0:
            sev = min(s / 4.0, 1.0)
            z[c]["scale"] = float(np.clip(z[c]["scale"] * (1 + 0.5 * sev), .5, 1.5))
            z[c]["gate"] = float(np.clip(z[c]["gate"] * (1 + 0.5 * sev), 0, 1))
            z[c]["conf"] = float(z[c]["conf"] * (1 - 0.3 * sev))
    return z

# ---------------------------------------------------------------- residual decoder
def features(Yhat, z):
    rows = []
    for y in Yhat:
        T, S, _ = split(y)
        m = T.mean()
        # component-wise "proposals" from the control signals, applied around the component's own level
        pT = z["trend"]["gate"] * z["trend"]["conf"] * ((z["trend"]["scale"] - 1) * (T - m) + z["trend"]["bias"])
        pS = z["seasonal"]["gate"] * z["seasonal"]["conf"] * ((z["seasonal"]["scale"] - 1) * S + z["seasonal"]["bias"])
        rows.append(np.stack([pT, pS], 1))
    return np.array(rows)                       # (n, H, 2)

def fit_decoder(Yhat, Y, z):
    F = features(Yhat, z).reshape(-1, 2)
    r = (Y - Yhat).reshape(-1)
    return np.linalg.lstsq(F, r, rcond=None)[0]  # learned weights w_T, w_S

def apply(Yhat, z, w):
    return Yhat + (features(Yhat, z) @ w)

# ---------------------------------------------------------------- run
Yhat_va = backbone(Xva)
d_val = diagnose(Yhat_va, Yva)
z0 = controller(d_val)
w = fit_decoder(Yhat_va, Yva, z0)

# held-out input statistics (cached) for shift detection
st_val = np.array([input_stats(x) for x in Xva])
mu, sd = st_val.mean(0), st_val.std(0) + 1e-9

Yhat_te = backbone(Xte)
st_te = np.array([input_stats(x) for x in Xte])
# one "calibration check": average the test batch's statistics, z-score against held-out
shift_scores = np.abs(st_te.mean(0) - mu) / sd
shift = {"trend": float(shift_scores[0]), "seasonal": float(shift_scores[1])}
z1 = adapt_controller(z0, shift)

pred = {"Backbone only": Yhat_te,
        "CTRL (no adaptation)": apply(Yhat_te, z0, w),
        "CTRL (full, with adaptation)": apply(Yhat_te, z1, w)}

def oracle(Yhat, Y):
    """Upper bound: the best possible re-scaling of the trend and seasonal parts, with no cap,
    chosen by peeking at the test answers. Real methods cannot do this."""
    rows = []
    for y in Yhat:
        T, S, _ = split(y)
        rows.append(np.stack([T - T.mean(), S], 1))
    F = np.array(rows).reshape(-1, 2)
    k = np.linalg.lstsq(F, (Y - Yhat).reshape(-1), rcond=None)[0]
    return Yhat + np.array(rows) @ k

pred["Oracle re-scaling (peeks at answers)"] = oracle(Yhat_te, Yte)

def mse(a, b): return float(np.mean((a - b) ** 2))
def mae(a, b): return float(np.mean(np.abs(a - b)))
res = {k: (mse(v, Yte), mae(v, Yte)) for k, v in pred.items()}

if QUIET:
    print("RESULT", *[f"{v[0]:.5f}" for v in res.values()], f"{shift['trend']:.3f}", f"{shift['seasonal']:.3f}")
    raise SystemExit
# ---------------------------------------------------------------- report
print("=== What the controller was shown (held-out) ===")
print(controller_prompt(d_val))
print("\n=== Control signals, held-out ===")
print(json.dumps(z0, indent=1))
print("\nlearned decoder weights [trend, seasonal]:", np.round(w, 3))
print("\n=== Shift scores (test vs held-out, input-only) ===", {k: round(v, 2) for k, v in shift.items()})
print("=== Adapted control signals ===")
print(json.dumps(z1, indent=1))
print("\n=== Test results ===")
base = res["Backbone only"][0]
for k, (m, a) in res.items():
    print(f"{k:32s} MSE {m:6.3f}  MAE {a:6.3f}  ({(1 - m / base) * 100:+5.1f}% MSE vs backbone)")
print("\nheld-out MSE: backbone %.3f -> CTRL %.3f" % (mse(Yhat_va, Yva), mse(apply(Yhat_va, z0, w), Yva)))

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"truth": "#222222", "base": "#c0504d", "ctrl0": "#e0a030", "ctrl": "#2a7fb8"}
idx = 20  # a representative test window
fig, ax = plt.subplots(figsize=(9, 4.2))
hx = np.arange(-48, 0); fx = np.arange(H)
ax.plot(hx, Xte[idx][-48:], color="#999999", lw=1.5, label="Recent history")
ax.plot(fx, Yte[idx], color=C["truth"], lw=2.2, label="Actual")
ax.plot(fx, pred["Backbone only"][idx], color=C["base"], lw=2, ls="--", label="Backbone only")
ax.plot(fx, pred["CTRL (no adaptation)"][idx], color=C["ctrl0"], lw=2, ls="-.", label="CTRL, no adaptation")
ax.plot(fx, pred["CTRL (full, with adaptation)"][idx], color=C["ctrl"], lw=2.4, label="CTRL (full)")
ax.axvline(0, color="#bbbbbb", lw=1)
ax.set_xlabel("Hours relative to forecast start"); ax.set_ylabel("Value")
ax.set_title("One forecast window after the regime change")
ax.legend(frameon=False, ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig("toy_forecast.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, (name, a), col in zip(axs, [("Held-out period (what the controller learned from)", bounds["val"][0] + 200),
                                     ("Test period (after the regime change)", bounds["test"][0] + 200)],
                              ["#2a7fb8", "#c0504d"]):
    ax.plot(np.arange(96), full[a:a + 96], color=col, lw=1.6)
    ax.set_title(name, fontsize=10); ax.set_xlabel("Hours")
axs[0].set_ylabel("Value")
fig.suptitle("Same trend, but the daily swing gets much bigger", fontsize=12)
fig.tight_layout(); fig.savefig("toy_series.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 3.8))
names = list(res); vals = [res[k][0] for k in names]
bars = ax.bar(["Backbone\nonly", "CTRL\n(no adaptation)", "CTRL\n(full)", "Oracle\n(upper bound)"], vals,
              color=[C["base"], C["ctrl0"], C["ctrl"], "#9aa5ad"])
for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
ax.set_ylabel("Test MSE (lower is better)"); ax.set_title("Forecast error after the regime change")
fig.tight_layout(); fig.savefig("toy_mse.png", dpi=160); plt.close(fig)
