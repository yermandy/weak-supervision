import numpy as np


def calc_prec_recall(X, y_true, y_pred=None, constrained=False):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    X = np.array(X)

    min_d = np.min(X)
    max_d = np.max(X)
    thresholds = np.linspace(min_d, max_d, min(200, len(X)))

    recall, prec = [], []

    y_true_sum = max(np.sum(y_true), 1e-8)

    for th in thresholds:

        if constrained and y_pred is not None:
            y_pred_th = (X <= th) & (y_pred == 1) 
        else:
            y_pred_th = X <= th

        y_pred_sum = max(np.sum(y_pred_th), 1e-8)

        tp = np.sum((y_true == y_pred_th) & (y_pred_th))
        # fp = np.sum((y_pred_th == 1) & (y_true != y_pred_th))
        # fn = np.sum((y_pred_th == 0) & (y_true != y_pred_th))
        
        r = tp / y_true_sum
        p = tp / y_pred_sum

        recall.append(r)
        prec.append(p)

    return recall, prec
