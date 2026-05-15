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
    run_pipeline(FM_dlib, "Method 1 — dlib ResNet-128")

    # ── Method 2: HOG────────────────────────────────────
    #print("\nExtracting HOG features …")
    #FM_hog = feature_matrix_hog(DATASET)
    #run_pipeline(FM_hog, "Method 2 — HOG")
