"""
频谱资源与信息率分析  —  Shi 2024 NeuroImage 核心框架复现

数据: Zenodo 8300517 preliminary sweep (10 subjects, 160 WN codes, single-target 注视)

本脚本直接调用 Zenodo 原始 core 代码 (vanilla 空间滤波器 + EEG2Code 反向 TRF)，
忠实复现论文的信息率上下界，并做频谱资源分解。

上界 (evoked-response SNR)：
    - vanilla 空间滤波，单试次 vs 试次平均残差
    - 偏差校正 (K-1)/K·SNR - 1/K
    - I = ∫₀^Nyquist log₂(1+SNR(f)) df   (神经谐波可超 30Hz)

下界 (stimulus reconstruction)：
    - EEG2Code 反向 TRF，单试次重建刺激 vs 重建残差
    - I = ∫₀^30Hz log₂(1+SNR(f)) df     (刺激本身带限于刷新率/2)

输出：
    figures/  fig01–fig05
    tables/   info_rate_bounds.csv, spectrum_data.csv, band_contributions.csv
"""
import sys, os, pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy.integrate import simpson
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──
DATASET = "D:/ProjData/datasets/cvep_broadband_white_noise_bci_zenodo8300517/extracted"
CORE    = os.path.join(DATASET, "core")
DATA_DIR = os.path.join(DATASET, "data/seperate/sweep")
STI_PATH = os.path.join(DATASET, "data/stimulation/sweep/STI.mat")
sys.path.insert(0, CORE)

# ── patch sklearn minmax_scale (paper code passes axis=-1) ──
import sklearn.preprocessing as _skpp
_orig_minmax = _skpp.minmax_scale
def _patched_minmax(X, feature_range=(0, 1), axis=0, copy=True):
    if axis == -1:
        axis = np.asarray(X).ndim - 1
    return _orig_minmax(X, feature_range=feature_range, axis=axis, copy=copy)
_skpp.minmax_scale = _patched_minmax

from spatialFilters import vanilla          # noqa: E402
from modeling import EEG2Code               # noqa: E402

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(OUT_DIR, "figures")
TAB_DIR  = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── parameters (paper Section 2.7) ──
SRATE = 250
WIN_LEN = 1.0
N_SAMPLES = int(SRATE * WIN_LEN)
REFRESH_RATE = 60
STI_CUTOFF = REFRESH_RATE // 2      # 30 Hz — stimulus bandlimit (lower bound)
TRF_TMIN, TRF_TMAX = -0.3, 0.0      # backward lag window
TRF_ALPHA = 0.95

CHN_NAMES = ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
             'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
             'O1', 'OZ', 'O2', 'CB1', 'CB2']


# ═════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════

def return_fft(s, srate=SRATE):
    s = s - np.mean(s, axis=-1, keepdims=True)
    N = s.shape[-1]
    A = fft(s, axis=-1)
    freqz = fftfreq(N, 1 / srate)[:N // 2]
    A = A.T[:N // 2]
    return freqz, A.T


def load_sti_250():
    """Load 160×60 STI, upsample to 250 Hz via frame-repeat (nearest)."""
    import scipy.io
    mat = scipy.io.loadmat(STI_PATH)
    sti = None
    for k, v in mat.items():
        if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim == 2:
            sti = v.astype(np.float64)
            break
    t60 = np.linspace(0, 1, sti.shape[1], endpoint=False)
    t250 = np.linspace(0, 1, N_SAMPLES, endpoint=False)
    out = np.zeros((sti.shape[0], N_SAMPLES))
    for i in range(sti.shape[0]):
        f = interp1d(t60, sti[i], kind="nearest", fill_value="extrapolate")
        out[i] = f(t250)
    return out


def load_subject(sid):
    """Return combined (X, y) across sessions. X:(n,21,250), y:(n,) 1-indexed."""
    with open(os.path.join(DATA_DIR, f"S_{sid}.pkl"), "rb") as fp:
        d = pickle.load(fp, encoding="latin1")
    Xs, ys = [], []
    n_sessions = d.shape[0]
    for si in range(n_sessions):
        chn = d[si, 3]
        idx = [chn.index(c) for c in CHN_NAMES]
        Xs.append(d[si, 1][:, idx, :N_SAMPLES])
        ys.append(d[si, 2])
    return np.concatenate(Xs, 0), np.concatenate(ys, 0), n_sessions


# ═════════════════════════════════════════════════════════════════
# per-subject bounds (paper's exact pipeline)
# ═════════════════════════════════════════════════════════════════

def upper_bound(X, y):
    """Evoked-response SNR upper bound (mutualINFO.py logic, integrate to Nyquist)."""
    X = X - np.mean(X, axis=-1, keepdims=True)
    classes = np.unique(y)

    model = vanilla(winLEN=WIN_LEN, lag=0, n_band=1)
    aveEvoked = np.squeeze(model.fit_transform(X, y))    # (n_class, T)
    xX = np.squeeze(model.transform(X))                   # (n_trial, T)
    aveEvoked = aveEvoked[:, :N_SAMPLES]
    xX = xX[:, :N_SAMPLES]

    aveEvoked_exp = np.concatenate([aveEvoked[classes == c] for c in y])
    xNoise = xX - aveEvoked_exp

    freqz, ss = return_fft(aveEvoked_exp)
    freqz, nn = return_fft(xNoise)
    sPower = (1 / (SRATE * N_SAMPLES)) * np.abs(ss) ** 2
    nPower = (1 / (SRATE * N_SAMPLES)) * np.abs(nn) ** 2

    ubSNR = [sPower[y == c].mean(0) / nPower[y == c].mean(0) for c in classes]
    Ks = list(Counter(y).values())
    logSNR = np.mean([np.log2(1 + ((K - 1) / K * snr - 1 / K))
                      for snr, K in zip(ubSNR, Ks)], axis=0)
    info = simpson(logSNR, x=freqz)                       # to Nyquist
    return info, freqz, logSNR, np.mean(ubSNR, axis=0)


def lower_bound(X, y, sti250):
    """Stimulus-reconstruction SNR lower bound (bounds.py logic, cutoff 30Hz)."""
    classes = np.unique(y)
    STI = np.stack([sti250[label - 1] for label in y])[:, :N_SAMPLES]

    model = EEG2Code(srate=SRATE, winLEN=WIN_LEN, tmin=TRF_TMIN, tmax=TRF_TMAX,
                     S=(sti250, classes), estimator=TRF_ALPHA, padding=True,
                     n_band=1, component=1)
    model.fit(X, y)
    sEst = model.predict(X)
    sN = STI - sEst

    freqz, STI_F  = return_fft(STI)
    freqz, sEst_F = return_fft(sEst)
    freqz, sN_F   = return_fft(sN)

    covSS = sEst_F * np.conjugate(sEst_F)
    covNN = sN_F * np.conjugate(sN_F)
    lbSNR = [covSS[y == c].mean(0) / covNN[y == c].mean(0) for c in classes]
    logSNR = np.mean([np.log2(1 + np.abs(snr)) for snr in lbSNR], axis=0)
    logSNR[freqz >= STI_CUTOFF] = 0
    info = simpson(logSNR, x=freqz)
    return info, freqz, logSNR, np.abs(model.trf)


# ═════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════

def main():
    print("Loading STI...")
    sti250 = load_sti_250()

    results = []
    for sid in range(1, 11):
        print(f"S{sid} ...", end=" ", flush=True)
        X, y, n_sess = load_subject(sid)
        ub, freqz, ub_log, ub_snr = upper_bound(X, y)
        lb, _, lb_log, trf = lower_bound(X, y, sti250)
        results.append(dict(sid=sid, n_sess=n_sess, ub=ub, lb=lb,
                            ub_log=ub_log, lb_log=lb_log, ub_snr=ub_snr,
                            trf=trf, freqz=freqz))
        print(f"UB={ub:.1f}  LB={lb:.1f}  ratio={ub/lb:.2f}")

    freqz = results[0]["freqz"]

    # ── Table 1: per-subject bounds ──
    rows = [dict(subject=f"S{r['sid']}", sessions=r["n_sess"],
                 ub_bps=round(r["ub"], 1), lb_bps=round(r["lb"], 1),
                 ratio=round(r["ub"] / r["lb"], 2)) for r in results]
    df = pd.DataFrame(rows)
    ub_m, ub_s = df.ub_bps.mean(), df.ub_bps.std()
    lb_m, lb_s = df.lb_bps.mean(), df.lb_bps.std()
    df.loc[len(df)] = ["Mean±SD", "-", f"{ub_m:.1f}±{ub_s:.1f}",
                       f"{lb_m:.1f}±{lb_s:.1f}", f"{ub_m/lb_m:.2f}"]
    df.to_csv(os.path.join(TAB_DIR, "info_rate_bounds.csv"), index=False)
    print(f"\nUpper: {ub_m:.1f}±{ub_s:.1f} bps (paper 63±20)")
    print(f"Lower: {lb_m:.1f}±{lb_s:.1f} bps (paper 25±3)")

    ub_snr_all = np.array([r["ub_snr"] for r in results])
    ub_log_all = np.array([r["ub_log"] for r in results])
    lb_log_all = np.array([r["lb_log"] for r in results])

    pd.DataFrame({
        "freq_hz": freqz,
        "ub_snr_mean": ub_snr_all.mean(0),
        "ub_log2_snr_mean": ub_log_all.mean(0),
        "ub_log2_snr_std": ub_log_all.std(0),
        "lb_log2_snr_mean": lb_log_all.mean(0),
        "lb_log2_snr_std": lb_log_all.std(0),
    }).to_csv(os.path.join(TAB_DIR, "spectrum_data.csv"), index=False)

    C_UB, C_LB = "#2c6fbb", "#e07b39"

    # ═══ Figure 1: SNR spectrum + spectral info density ═══
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    snr_db = 10 * np.log10(ub_snr_all.mean(0) + 1e-30)
    ax.plot(freqz, snr_db, color=C_UB, lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.6)
    ax.axvline(STI_CUTOFF, color="gray", ls=":", lw=0.8)
    ax.annotate("stimulus bandlimit\n(refresh/2 = 30 Hz)", xy=(30, snr_db.max()),
                xytext=(35, snr_db.max() - 2), fontsize=8, color="gray")
    ax.set_ylabel("Evoked SNR (dB)")
    ax.set_title("(a) Evoked-response SNR spectrum (spatial-filtered, upper-bound)")
    ax.set_xlim(0, 60)

    ax = axes[1]
    ub_m_log, ub_sd = ub_log_all.mean(0), ub_log_all.std(0)
    lb_m_log, lb_sd = lb_log_all.mean(0), lb_log_all.std(0)
    ax.fill_between(freqz, ub_m_log - ub_sd, ub_m_log + ub_sd, color=C_UB, alpha=0.15)
    ax.fill_between(freqz, lb_m_log - lb_sd, lb_m_log + lb_sd, color=C_LB, alpha=0.15)
    ax.plot(freqz, ub_m_log, color=C_UB, lw=1.5, label=f"Upper bound ({ub_m:.0f} bps)")
    ax.plot(freqz, lb_m_log, color=C_LB, lw=1.5, label=f"Lower bound ({lb_m:.0f} bps)")
    ax.axvline(STI_CUTOFF, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log₂(1+SNR(f))  (bits/s/Hz)")
    ax.set_title("(b) Spectral information density")
    ax.legend()
    ax.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig01_snr_spectrum.png"), dpi=200)
    plt.close(fig)
    print("fig01 saved")

    # ═══ Figure 2: cumulative info rate + per-subject bounds ═══
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for r in results:
        ub_cum = np.array([simpson(r["ub_log"][:n], x=freqz[:n]) if n > 1 else 0
                           for n in range(1, len(freqz) + 1)])
        lb_cum = np.array([simpson(r["lb_log"][:n], x=freqz[:n]) if n > 1 else 0
                           for n in range(1, len(freqz) + 1)])
        ax.plot(freqz, ub_cum, color=C_UB, alpha=0.25, lw=0.8)
        ax.plot(freqz, lb_cum, color=C_LB, alpha=0.25, lw=0.8)
    ub_cum_m = np.array([simpson(ub_log_all.mean(0)[:n], x=freqz[:n]) if n > 1 else 0
                         for n in range(1, len(freqz) + 1)])
    lb_cum_m = np.array([simpson(lb_log_all.mean(0)[:n], x=freqz[:n]) if n > 1 else 0
                         for n in range(1, len(freqz) + 1)])
    ax.plot(freqz, ub_cum_m, color=C_UB, lw=2.2, label=f"UB mean {ub_m:.0f} bps")
    ax.plot(freqz, lb_cum_m, color=C_LB, lw=2.2, label=f"LB mean {lb_m:.0f} bps")
    ax.set_xlabel("Frequency cutoff (Hz)")
    ax.set_ylabel("Cumulative info rate (bits/s)")
    ax.set_title("(a) Cumulative I(f) = ∫₀ᶠ log₂(1+SNR) df'")
    ax.legend()
    ax.set_xlim(0, 125)

    ax = axes[1]
    sids = [f"S{r['sid']}" for r in results]
    x = np.arange(len(sids))
    ax.bar(x - 0.2, [r["ub"] for r in results], 0.38, color=C_UB, alpha=0.85, label="Upper")
    ax.bar(x + 0.2, [r["lb"] for r in results], 0.38, color=C_LB, alpha=0.85, label="Lower")
    ax.axhline(63, color=C_UB, ls="--", lw=0.8, alpha=0.6)
    ax.axhline(25, color=C_LB, ls="--", lw=0.8, alpha=0.6)
    ax.text(len(sids) - 0.5, 64, "paper 63", fontsize=8, color=C_UB)
    ax.text(len(sids) - 0.5, 26, "paper 25", fontsize=8, color=C_LB)
    ax.set_xticks(x); ax.set_xticklabels(sids, rotation=45)
    ax.set_ylabel("Information rate (bits/s)")
    ax.set_title("(b) Per-subject bounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig02_cumulative_info_rate.png"), dpi=200)
    plt.close(fig)
    print("fig02 saved")

    # ═══ Figure 3: backward TRF kernel + frequency response ═══
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lag_ms = np.linspace(TRF_TMIN, TRF_TMAX, len(results[0]["trf"])) * 1000
    trf_all = np.array([r["trf"] / (np.abs(r["trf"]).max() + 1e-30) for r in results])
    ax = axes[0]
    for t in trf_all:
        ax.plot(lag_ms, t, color="#4a9b6e", alpha=0.3, lw=0.8)
    ax.plot(lag_ms, trf_all.mean(0), color="#2e6b4a", lw=2.2, label="Grand mean")
    ax.axhline(0, color="gray", ls="--", lw=0.6)
    ax.set_xlabel("Lag τ (ms)")
    ax.set_ylabel("TRF amplitude (norm.)")
    ax.set_title("(a) Backward TRF kernel h(τ)")
    ax.legend()

    ax = axes[1]
    trf_mean = np.array([r["trf"] for r in results]).mean(0)
    H = np.abs(fft(trf_mean, n=N_SAMPLES))[:N_SAMPLES // 2]
    H = H / (H.max() + 1e-30)
    fr = fftfreq(N_SAMPLES, 1 / SRATE)[:N_SAMPLES // 2]
    ax.plot(fr, H, color="#2e6b4a", lw=1.5)
    ax.axvline(STI_CUTOFF, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H(f)| (norm.)")
    ax.set_title("(b) TRF frequency response")
    ax.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig03_trf_kernel.png"), dpi=200)
    plt.close(fig)
    print("fig03 saved")

    # ═══ Figure 4: frequency-band contributions ═══
    bands = {"δ 1-4": (1, 4), "θ 4-8": (4, 8), "α 8-13": (8, 13),
             "β 13-30": (13, 30), ">30 (harm.)": (30, 125)}
    fig, ax = plt.subplots(figsize=(9, 5))
    rows_b = []
    ub_means, ub_stds, lb_means, lb_stds = [], [], [], []
    for bn, (lo, hi) in bands.items():
        m = (freqz >= lo) & (freqz < hi)
        ubb = [simpson(r["ub_log"][m], x=freqz[m]) for r in results]
        lbb = [simpson(r["lb_log"][m], x=freqz[m]) for r in results]
        ub_means.append(np.mean(ubb)); ub_stds.append(np.std(ubb))
        lb_means.append(np.mean(lbb)); lb_stds.append(np.std(lbb))
        rows_b.append(dict(band=bn, ub_mean=round(np.mean(ubb), 1),
                           ub_std=round(np.std(ubb), 1),
                           lb_mean=round(np.mean(lbb), 1),
                           lb_std=round(np.std(lbb), 1),
                           ub_pct=round(np.mean(ubb) / ub_m * 100, 1),
                           lb_pct=round(np.mean(lbb) / lb_m * 100, 1)))
    pd.DataFrame(rows_b).to_csv(os.path.join(TAB_DIR, "band_contributions.csv"), index=False)
    xb = np.arange(len(bands))
    ax.bar(xb - 0.2, ub_means, 0.38, yerr=ub_stds, capsize=3, color=C_UB, alpha=0.85, label="Upper")
    ax.bar(xb + 0.2, lb_means, 0.38, yerr=lb_stds, capsize=3, color=C_LB, alpha=0.85, label="Lower")
    ax.set_xticks(xb); ax.set_xticklabels(list(bands.keys()))
    ax.set_ylabel("Band contribution (bits/s)")
    ax.set_title("Frequency-band contributions to information rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig04_band_contributions.png"), dpi=200)
    plt.close(fig)
    print("fig04 saved")

    # ═══ Figure 5: spectrum resource occupancy (WN measured + SSVEP band footprint) ═══
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(freqz, 0, ub_m_log, color=C_UB, alpha=0.30,
                    label="WN-CDMA measured info density (this data)")
    ax.plot(freqz, ub_m_log, color=C_UB, lw=1.6)

    # SSVEP-FDMA spectral footprint: fundamentals span 8–15.8 Hz (Benchmark),
    # 2nd harmonics 16–31.6 Hz. Shade the occupied bands (footprint, not amplitude).
    ax.axvspan(8.0, 15.8, color="#b5477f", alpha=0.18)
    ax.axvspan(16.0, 31.6, color="#b5477f", alpha=0.08)
    ymax = ax.get_ylim()[1]
    ax.text(11.9, ymax * 0.92, "SSVEP\nfundamentals", ha="center", va="top",
            fontsize=8.5, color="#8f3563")
    ax.text(23.8, ymax * 0.62, "2nd harm.", ha="center", va="top",
            fontsize=8, color="#8f3563", alpha=0.8)

    # bracket showing WN occupancy
    wn_hi = freqz[np.where(ub_m_log > 0.4)[0].max()]
    ax.annotate("", xy=(1, ymax * 0.05), xytext=(wn_hi, ymax * 0.05),
                arrowprops=dict(arrowstyle="<->", color=C_UB, lw=1.3))
    ax.text((1 + wn_hi) / 2, ymax * 0.08,
            f"WN occupies ~1–{wn_hi:.0f} Hz + harmonics",
            ha="center", fontsize=8.5, color=C_UB)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log₂(1+SNR(f))  (bits/s/Hz)")
    ax.set_title("Spectrum resource occupancy: broadband WN-CDMA vs narrowband SSVEP-FDMA footprint")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig05_fdma_vs_cdma_spectrum.png"), dpi=200)
    plt.close(fig)
    print("fig05 saved")

    print(f"\nDone. Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
