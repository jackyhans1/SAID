#!/usr/bin/env python
# temp_scaling.py
import numpy as np

def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def find_best_T(logits, labels, Tmin=0.5, Tmax=5.0, step=0.05):
    """
    logits : (N, C) numpy array (softmax 미적용)
    labels : (N,)   numpy array (int, 0/1)
    반환값  : best temperature (float)
    """
    best_T, best_nll = 1.0, float("inf")
    for T in np.arange(Tmin, Tmax + 1e-9, step):
        prob = softmax_np(logits / T)
        nll  = -np.mean(np.log(prob[np.arange(len(labels)), labels] + 1e-12))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T
