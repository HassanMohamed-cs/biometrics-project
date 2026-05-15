import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def Calculategenimp(
    FM: ArrayLike,
    S: int,
    Tr: int = 10,
    Ts: int = 2,
    samples_per_user: int = 13,
    title: str = "Genuine vs. Impostor Cosine Similarity Distributions",
) -> tuple[np.ndarray, np.ndarray]:
    FM = np.array(FM, dtype=np.float64)
    Gen: list[float] = []
    Imp: list[float] = []

    for i in range(S):
        start = i * samples_per_user

        # Gallery (training) and probe (test) column indices
        train_idx = range(start, start + Tr)
        test_idx  = range(start + Tr, start + Tr + Ts)

        train_vecs = FM[:, train_idx]            # (D, Tr)
        test_vecs  = FM[:, test_idx]             # (D, Ts)

        # L2-normalise gallery columns once
        train_norms = np.linalg.norm(train_vecs, axis=0, keepdims=True)
        train_norms[train_norms < 1e-10] = 1.0  # avoid division by zero
        train_norm  = train_vecs / train_norms   # (D, Tr)

        for t in range(Ts):
            probe = test_vecs[:, t : t + 1]     # (D, 1)
            p_norm_val = np.linalg.norm(probe)
            if p_norm_val < 1e-10:
                continue
            probe_norm = probe / p_norm_val      # (D, 1)

            # ── Genuine: probe vs. same-subject gallery ──────────────────────
            sims_gen = float(np.max(np.sum(train_norm * probe_norm, axis=0)))
            Gen.append(sims_gen)

            # ── Impostor: probe vs. every other subject's gallery ────────────
            for other_s in range(S):
                if other_s == i:
                    continue
                o_start = other_s * samples_per_user
                o_train_idx = range(o_start, o_start + Tr)
                o_train_vecs  = FM[:, o_train_idx]
                o_norms       = np.linalg.norm(o_train_vecs, axis=0, keepdims=True)
                o_norms[o_norms < 1e-10] = 1.0
                o_train_norm  = o_train_vecs / o_norms

                sim_imp = float(np.max(np.sum(o_train_norm * probe_norm, axis=0)))
                Imp.append(sim_imp)

    Gen_arr = np.array(Gen)
    Imp_arr = np.array(Imp)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.hist(Gen_arr, bins=50, alpha=0.6, label="Genuine", color="steelblue",
             density=True)
    plt.hist(Imp_arr, bins=50, alpha=0.6, label="Impostor", color="crimson",
             density=True)
    plt.title(title)
    plt.xlabel("Matching Score (Cosine Similarity)")
    plt.ylabel("Normalised Frequency")
    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return Gen_arr, Imp_arr
