import numpy as np
from Features   import feature_matrix_dlib, feature_matrix_hog
from Gen_Imp    import Calculategenimp
from Metric    import metric_calc, identification_metrics
DATASET    = "splited_dataset"
S          = 50
TR         = 10
TS         = 2
SPU        = 13
def run_pipeline(FM: np.ndarray, method_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")

    print("\n[1] Computing Genuine / Impostor scores …")
    gen, imp = Calculategenimp(FM, S=S, Tr=TR, Ts=TS,
                               samples_per_user=SPU,
                               title=f"Gen-Imp — {method_name}")

    print("\n[2] Verification metrics …")
    metric_calc(gen, imp)

    print("\n[3] Identification metrics …")
    identification_metrics(FM, S=S, Tr=TR,
                           samples_per_user=SPU,
                           threshold= 0.01)


if __name__ == "__main__":
    # ── Method 1: dlib ResNet-128──────────────────────
    print("Extracting dlib features …")
    #FM_dlib = feature_matrix_dlib(DATASET)
    FM_dlib = np.load("FM_DLIB.npy")
    num_subjects = 50
    samples_per_subject = 13

    # Build new column order
    new_order = []
    for subj in range(num_subjects):
        start = subj * samples_per_subject
        end = start + samples_per_subject

        # indices for this subject
        subject_indices = list(range(start, end))

        # train = samples 4–13 (1-based) → indices 3–12
        train_indices = subject_indices[3:13]
        identification_indices = subject_indices[0:1]
        # others = samples 1–3 (1-based) → indices 0–2
        other_indices = subject_indices[1:3]

        # append train first, then others
        new_order.extend(train_indices + other_indices + identification_indices)

    # Reorder along axis=1
    features_reordered = FM_dlib[:, new_order]

    print(features_reordered.shape)  # (128, 650)
    run_pipeline(features_reordered, "Method 1 — dlib ResNet-128")

    # ── Method 2: HOG────────────────────────────────────
    print("\nExtracting HOG features …")
    FM_hog = np.load("FM_HOG.npy")
    for subj in range(num_subjects):
        start = subj * samples_per_subject
        end = start + samples_per_subject
        subject_indices = list(range(start, end))
        train_indices = subject_indices[3:13]
        identification_indices = subject_indices[0:1]
        other_indices = subject_indices[1:3]
        new_order.extend(train_indices + other_indices + identification_indices)
    features_reordered = FM_hog[:, new_order]
    print(features_reordered.shape)
    run_pipeline(features_reordered, "Method 2 — HOG")
