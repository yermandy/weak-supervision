import numpy as np


def calc_prec_recall(y_true, preds, dists):

    y_true = np.array(y_true)
    preds = np.array(preds)
    dists = np.array(dists)

    min_d = np.min(dists)
    max_d = np.max(dists)
    thresholds = np.linspace(min_d, max_d, min(200, len(dists)))

    recall, prec = [], []

    y_true_sum = np.sum(y_true == 1) + 1e-8

    for th in thresholds:

        y_pred = np.where((dists <= th), 1, 0)
        # y_pred = np.where((dists <= th) & (preds == 1), 1, 0)

        y_pred_sum = np.sum(y_pred == 1) + 1e-8

        tp = np.sum((y_true == y_pred) & (y_pred == 1))
        # fp = np.sum((y_pred == 1) & (y_true != y_pred))
        # fn = np.sum((y_pred == 0) & (y_true != y_pred))
        
        r = tp / y_true_sum
        p = tp / y_pred_sum

        recall.append(r)
        prec.append(p)

    return recall, prec


'''
def calc_prec_recall_per_id(y_true, preds, dists, subjects):
    
    y_true = np.array(y_true)
    preds = np.array(preds)
    dists = np.array(dists)

    min_d = np.min(dists) + 1e-1
    max_d = np.max(dists)
    thresholds = np.linspace(min_d, max_d, min(200, len(dists)))

    recall, prec = [], []

    y_true_sum = np.sum(y_true == 1)

    # for s in subjects:
    #     y_true_sum += np.sum(y_true[s] == 1)

    for th in thresholds:

        # y_pred = np.where((dists <= th), 1, 0)
        y_pred = np.where((dists <= th) & (preds == 1), 1, 0)

        tp = 0
        y_pred_sum = 0

        r = 0
        p = 0

        for s in subjects:
            y_true_s = y_true[s]
            y_pred_s = y_pred[s]
            
            tp = np.sum((y_true_s == y_pred_s) & (y_pred_s == 1))

            r += tp / (np.sum(y_true_s == 1) + 1e-8)
            p += tp / (np.sum(y_pred_s == 1) + 1e-8)

            ## Check
            # tp += np.sum((y_true_s == y_pred_s) & (y_pred_s == 1))
            # y_pred_sum += np.sum(y_pred_s == 1)
            
        # print(x)
        # print(y)
        # print(len(subjects))
        # print()

        r = r / len(subjects)
        p = p / len(subjects)

        ## Check
        # r = tp / (y_true_sum + 1e-8)
        # p = tp / (y_pred_sum + 1e-8)

        recall.append(r)
        prec.append(p)

    return recall, prec
'''