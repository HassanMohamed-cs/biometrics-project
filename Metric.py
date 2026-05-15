import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


# ═══════════════════════════════════════════════════════════════════════════════
# Verification metrics
# ═══════════════════════════════════════════════════════════════════════════════

def fmr_fnmr(
    gen: ArrayLike,
    imp: ArrayLike,
    T: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    gen = np.asarray(gen, dtype=float)
    imp = np.asarray(imp, dtype=float)

    S_min = min(gen.min(), imp.min())
    S_max = max(gen.max(), imp.max())
    step  = (S_max - S_min) / (T - 1)

    fmr, fnmr, tmr = [], [], []
    for t in range(T):
        thresh  = S_min + t * step
        fmr_i   = np.mean(imp > thresh)
        tmr_i   = np.mean(gen > thresh)
        fnmr_i  = 1.0 - tmr_i
        fmr.append(fmr_i)
        fnmr.append(fnmr_i)
        tmr.append(tmr_i)

    fmr  = np.array(fmr)
    fnmr = np.array(fnmr)
    tmr  = np.array(tmr)

    # ── ROC plot ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(6, 6))
    plt.plot(fmr, tmr, color="steelblue", lw=2)
    plt.xlabel("False Match Rate (FMR)")
    plt.ylabel("True Match Rate (TMR)")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fmr, fnmr, tmr


def metric_calc(gen: ArrayLike, imp: ArrayLike) -> dict:
    gen = np.asarray(gen, dtype=float)
    imp = np.asarray(imp, dtype=float)

    # D-prime
    d_prime = (
        (2 ** 0.5)
        * np.abs(np.mean(gen) - np.mean(imp))
        / np.sqrt(np.var(gen) + np.var(imp) + 1e-12)
    )

    fmr, fnmr, tmr = fmr_fnmr(gen, imp)

    # EER  (threshold where FMR ≈ FNMR)
    diffs    = np.abs(fmr - fnmr)
    min_idx  = int(np.argmin(diffs))
    eer      = float((fmr[min_idx] + fnmr[min_idx]) / 2)

    # TMR @ FMR = 1 %
    idx_1    = int(np.argmin(np.abs(fmr - 0.01)))
    tmr_1    = float(tmr[idx_1])

    # TMR @ FMR = 0.01 %
    idx_001  = int(np.argmin(np.abs(fmr - 0.0001)))
    tmr_001  = float(tmr[idx_001])

    print("─" * 55)
    print(f"  EER                        : {eer:.4f}  ({eer*100:.2f} %)")
    print(f"  D-prime                    : {d_prime:.4f}")
    print(f"  TMR @ FMR = 1 %            : {tmr_1:.4f}  ({tmr_1*100:.2f} %)")
    print(f"  TMR @ FMR = 0.01 %         : {tmr_001:.4f}  ({tmr_001*100:.2f} %)")
    print("─" * 55)

    return dict(d_prime=d_prime, eer=eer, tmr_at_fmr1=tmr_1, tmr_at_fmr001=tmr_001)

def identification_metrics(
    FM: ArrayLike,
    S: int,
    Tr: int = 10,
    samples_per_user: int = 13,
    threshold: float | None = None,
) -> dict:

    FM = np.array(FM, dtype=np.float64)

    # Build per-subject gallery matrices and normalise
    galleries: list[np.ndarray] = []
    for s in range(S):
        start = s * samples_per_user
        g = FM[:, start : start + Tr]
        norms = np.linalg.norm(g, axis=0, keepdims=True)
        norms[norms < 1e-10] = 1.0
        galleries.append(g / norms)          # (D, Tr)

    correct = 0
    total_genuine = 0

    for i in range(S):
        start     = i * samples_per_user
        probe = FM[:, start + 12]
        p_norm = np.linalg.norm(probe)
        if p_norm < 1e-10:
            continue
        probe_norm = probe / p_norm      # (D,)

            # Cosine similarity to each subject's gallery (max pooling)
        scores = np.array([
            float(np.max(g.T @ probe_norm))
            for g in galleries
        ])                               # (S,)

        predicted = int(np.argmax(scores))
        best_score = scores[predicted]

            # ── Rank-1 accuracy ───────────────────────────────────────────────
        total_genuine += 1
        if predicted == i:
            correct += 1


    rank1 = correct / total_genuine if total_genuine > 0 else 0.0

    print("─" * 55)
    print(f"  Rank-1 Accuracy (TPIR)     : {rank1:.4f}  ({rank1*100:.2f} %)")

    return dict(rank1=rank1)
