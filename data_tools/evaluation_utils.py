'''
Methods for evaluating the predictions of models written to .json files
in the officail format.
'''
import json
import subprocess

from classif_experim.classif_utils import classif_scores
from data_tools.dataset_utils import BINARY_MAPPING_CRITICAL_POS, BINARY_MAPPING_CONSPIRACY_POS
from evaluation import oppositional_evaluator

def evaluate_classif_predictions(pred_file: str, gold_file: str, positive_class: str) -> None:
    """
    Evaluate the predictions in the pred_file against the gold_file.

    Args:
        pred_file (str): Path to the .json file with the predictions.
        gold_file (str): Path to the .json file with the gold labels.
        positive_class (str): The positive class label ('conspiracy' or 'critical').

    Returns:
        None
    """

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

def run_official_evaluation_script(task: str, predictions: str, gold: str, outdir: str = '.') -> None:
    """
    Run the official evaluation script with the given arguments.

    Args:
        task (str): The evaluation task.
        predictions (str): Path to the predictions file.
        gold (str): Path to the gold labels file.
        outdir (str, optional): Directory to save the output. Default is '.'.

    Returns:
        None
    """

    eval_script_path = oppositional_evaluator.__file__
    command = ['python', eval_script_path, task, '--predictions', predictions, '--gold', gold, '--outdir', outdir]
    process = subprocess.run(command, text=True, capture_output=True)
    # Print the output
    print("Eval. Script Output:", process.stdout, "\n")
    if process.stderr:
        print("Eval. Script Errors:", process.stderr)
