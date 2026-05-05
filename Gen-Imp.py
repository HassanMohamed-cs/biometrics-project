import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import auc

def Calculategenimp(FM: ArrayLike, S: int, Tr: int = 11, Ts: int = 3):
    FM = np.array(FM)
    Gen = []
    Imp = []
    samplesperuser = 14

    for i in range(S):
        start = i * samplesperuser
        train_idx = range(start, start + Tr)
        test_idx = range(start + Tr, start + Tr + Ts)
        train_vectors = FM[:, train_idx]
        test_vectors = FM[:, test_idx]

        for t in range(Ts):
            test_vec = test_vectors[:, t:t + 1]
            train_norm = train_vectors / np.linalg.norm(train_vectors, axis=0, keepdims=True)
            test_norm = test_vec / np.linalg.norm(test_vec, axis=0, keepdims=True)
            sims = np.sum(train_norm * test_norm, axis=0)
            Gen.extend(sims)

            for other_s in range(S):
                if other_s == i:
                    continue
                o_start = other_s * samplesperuser
                o_train_idx = range(o_start, o_start + Tr)
                o_train_vectors = FM[:, o_train_idx]
                o_train_norm = o_train_vectors / np.linalg.norm(o_train_vectors, axis=0, keepdims=True)
                test_norm = test_vec / np.linalg.norm(test_vec, axis=0, keepdims=True)
                sims = np.sum(o_train_norm * test_norm, axis=0)
                Imp.extend(sims)

    plt.hist(Gen, bins=30, alpha=0.5, label='Group A', color='blue')
    plt.hist(Imp, bins=30, alpha=0.5, label='Group B', color='red')
    plt.title('Genuine vs. Impostor Cosine Similarity Distributions')
    plt.xlabel('Matching Score')
    plt.ylabel('Normalize Frequency')
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    return np.array(Gen), np.array(Imp)





