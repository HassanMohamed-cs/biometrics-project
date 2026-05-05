import numpy as np
import matplotlib.pyplot as plt
def metric_calc (gen, imp):
    d_prime = ((2**0.5) * np.abs(np.mean(gen) - np.mean(imp))) / ((np.var(gen) + np.var(imp)) ** 0.5)
    fmr, fnmr, tmr = fmr_fnmr(gen, imp)
    min_diff = np.abs(fmr[0]-fnmr[0])
    min_diff_i = 0
    for i in range(1, len(fnmr)):
        if np.abs(fmr[i] - fnmr[i]) < min_diff:
            min_diff_i = i
    eer = (fmr[min_diff_i] + fnmr[min_diff_i]) / 2
    t = 0.01
    tmr_i = np.argmin(np.abs(fmr - t))
    tmr_t1 = tmr[tmr_i]
    t = 0.0001
    tmr_i = np.argmin(np.abs(fmr - t))
    tmr_t2 = tmr[tmr_i]
    print(f'EER: {eer} \t D-prim: {d_prime} \n TMR at FMR = 1% : {tmr_t1} \t TMR at FMR = 0.01% : {tmr_t2}')
def fmr_fnmr(gen, imp):
    T = 150
    fmr, fnmr, tmr = [], [], []
    S_min = min(np.min(gen), np.min(imp))
    S_max = max(np.max(gen), np.max(imp))
    p = (S_max - S_min) / (T-1)
    for t in range(T):
        T_i = S_min + t * p
        fmr_i = np.mean(imp > T_i)
        tmr_i = np.mean(gen > T_i)
        fnmr_i = 1 - tmr_i
        fmr.append(fmr_i)
        fnmr.append(fnmr_i)
        tmr.append(tmr_i)
    fmr = np.array(fmr)
    fnmr = np.array(fnmr)
    tmr = np.array(tmr)
    plt.figure(figsize=(6,6))
    plt.plot(fmr, tmr, color='blue', lw=2)
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('True Match Rate (TMR)')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()
    return fmr, fnmr, tmr




