
import torch, numpy as np, matplotlib.pyplot as plt, os, sklearn.metrics as skm
from sklearn.metrics import recall_score, f1_score

def calc_class_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(float) / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def accuracy(preds, trues):
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    return (preds == trues).mean()

def uar(preds, trues):
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    # macro recall == UAR
    return recall_score(trues, preds, average="macro")

def f1(preds, trues):
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    return f1_score(trues, preds, average="macro")

def plot_metric(curves, names, ylabel, save_path):
    plt.figure()
    for y, n in zip(curves, names):
        plt.plot(y, label=n)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(trues, preds, save_path):
    cm = skm.confusion_matrix(trues, preds)
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
