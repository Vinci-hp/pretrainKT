from sklearn.metrics import roc_auc_score, accuracy_score


def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

