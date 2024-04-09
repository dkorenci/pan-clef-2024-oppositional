from functools import partial

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef

f1_macro = partial(f1_score, average='macro')
f1_binary = partial(f1_score, average='binary')

def f1_score_negative_class(y_true, y_pred):
    """ Calculate the F1 score for the negative class in a binary classification setting. """
    assert set(y_true).issubset({0, 1})
    assert set(y_pred).issubset({0, 1})
    # Inverting the labels: 0 becomes 1 and 1 becomes 0
    y_true_inverted = [1 if label == 0 else 0 for label in y_true]
    y_pred_inverted = [1 if label == 0 else 0 for label in y_pred]
    return f1_score(y_true_inverted, y_pred_inverted, average='binary')

def classif_scores(setup='binary'):
    f1_macro = partial(f1_score, average='macro')
    f1_binary = partial(f1_score, average='binary')
    if setup == 'binary':
        score_fns = {'F1': f1_binary, 'ACC': accuracy_score,
                     'prec': precision_score, 'recall': recall_score}
    elif setup == 'multiclass':
        score_fns = { 'F1_macro': f1_macro, 'ACC':accuracy_score, 'MCC': matthews_corrcoef }
    elif setup == 'all':
        score_fns = {'F1_macro': f1_macro, 'F1': f1_binary, 'F1-neg': f1_score_negative_class,
                     'ACC': accuracy_score, 'prec': precision_score, 'recall': recall_score, 'MCC': matthews_corrcoef}
    elif setup == 'span-binary':
        score_fns = {'F1m': f1_macro, 'F1': f1_binary, 'P': precision_score, 'R': recall_score}
    return score_fns
