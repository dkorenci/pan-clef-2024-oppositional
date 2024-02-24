'''
Methods for evaluating the predictions of models written to .json files
in the officail format.
'''
import json

from classif_experim.classif_utils import classif_scores
from data_tools.dataset_utils import BINARY_MAPPING_CRITICAL_POS, BINARY_MAPPING_CONSPIRACY_POS


def evaluate_classif_predictions(pred_file: str, gold_file: str, positive_class: str):
    '''
    Evaluate the predictions in the pred_file against the gold_file.
    :param pred_file: .json file with the predictions
    :param gold_file: .json file with the gold labels
    :param positive_class: 'conspiracy' or 'critical'
    :return:
    '''
    with open(pred_file, 'r', encoding='utf-8') as file:
        pred_data = json.load(file)
    with open(gold_file, 'r', encoding='utf-8') as file:
        gold_data = json.load(file)
    if len(pred_data) != len(gold_data):
        raise ValueError(f'Length of predictions ({len(pred_data)}) does not match length of gold data ({len(gold_data)})')
    # check that the 'id' fields match exactly, disregaridng the order of the documents
    pred_ids = sorted([doc['id'] for doc in pred_data])
    gold_ids = sorted([doc['id'] for doc in gold_data])
    if pred_ids != gold_ids:
        raise ValueError(f'Ids do not match: {pred_ids} != {gold_ids}')
    # align predictions and gold data by id
    pred_labels = [doc['category'] for doc in pred_data]
    gold_labels = [doc['category'] for doc in gold_data]
    # map to binary integers
    positive_class = positive_class.lower()
    if positive_class == 'conspiracy': binmap = BINARY_MAPPING_CONSPIRACY_POS
    elif positive_class == 'critical': binmap = BINARY_MAPPING_CRITICAL_POS
    else: raise ValueError(f'Unknown positive class: {positive_class}')
    pred_labels = [binmap[label] for label in pred_labels]
    gold_labels = [binmap[label] for label in gold_labels]
    score_fns = classif_scores('all')
    scores_fmtd = "\n".join([f"{fname:10}: {f(gold_labels, pred_labels):.3f}" for fname, f in score_fns.items()])
    print(scores_fmtd)
