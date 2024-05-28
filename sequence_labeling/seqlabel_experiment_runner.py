import argparse
import datetime
import logging
import os
import time
from copy import copy
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from classif_experim import classif_experiment_runner
from classif_experim.classif_experiment_runner import setup_logging
from classif_experim.classif_utils import classif_scores
from classif_experim.pynvml_helpers import print_cuda_devices
from data_tools.dataset_loaders import load_dataset_full
from data_tools.spacy_utils import get_doc_id, get_doc_class, get_annoation_tuples_from_doc
from evaluation.oppositional_evaluator import calc_macro_averages
from sequence_labeling.seqlab_sklearn_wrapper_multitask import OppSequenceLabelerMultitask
from sequence_labeling.span_f1_metric import compute_score_pr
from data_tools.span_data_definitions import SPAN_LABELS_OFFICIAL

global logger

def task_label_data() -> Tuple[List[str], Dict[str, int]]:
    """
    Define variables with information on the task labels. Each task corresponds to a single span label,
    since the model is multitask and each task is a sequence labeling task for a single label.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing the list of task labels and a dictionary mapping task labels to task indices.
    """

    task_labels = sorted(list(SPAN_LABELS_OFFICIAL.values()))
    task_indices = {l: i for i, l in enumerate(task_labels)}
    return task_labels, task_indices

def run_crossvalid_seqlab_transformers(lang: str, model_label: str, model_params: Dict, src_langs: List[List[str]], num_folds: int = 5, 
                                       rnd_seed: int = 3154561, test: Union[int, bool] = False, pause_after_fold: int = 0) -> None:
    """
    Run x-fold cross-validation for a given model, and report the results.

    Args:
        lang (str): Language code.
        model_label (str): Model label.
        model_params (Dict): Dictionary of model parameters.
        num_folds (int, optional): Number of folds for cross-validation. Default is 5.
        rnd_seed (int, optional): Random seed for reproducibility. Default is 3154561.
        test (Union[int, bool], optional): Number of train examples to use for testing, or False to use all data. Default is False.
        pause_after_fold (int, optional): Pause duration in minutes after each fold. Default is 0.

    Returns:
        None
    """

    logger.info(f'RUNNING crossvalid. for model: {model_label}')
    docs = load_dataset_full(lang, format='docbin', src_langs=src_langs)
    if test: docs = docs[:test]
    foldgen = StratifiedKFold(n_splits=num_folds, random_state=rnd_seed, shuffle=True)
    fold_index = 0
    task_labels, task_indices = task_label_data()
    # make columns list that has values 'P', 'R', 'F1' and f'{X}-F1', f'{X}-P', f'{X}-R' for all X in TASK_LABELS
    columns = ['macro-F1', 'macro-P', 'macro-R'] + [f'{X}-F1' for X in task_labels] + [f'{X}-P' for X in task_labels] + [f'{X}-R' for X in task_labels]
    results_df = pd.DataFrame(columns=columns)
    rseed = rnd_seed
    classes = [get_doc_class(doc) for doc in docs]
    for train_index, test_index in foldgen.split(docs, classes):
        logger.info(f'Starting Fold {fold_index+1}')
        model = build_seqlab_model(model_label, rseed, model_params, task_labels, task_indices)
        logger.info(f'model built')
        # split data using the indices (these are not numpy or pandas arrays, so we can't use them directly)
        docs_train, docs_test = [], []
        for i in train_index: docs_train.append(docs[i])
        for i in test_index: docs_test.append(docs[i])
        # train model
        model.fit_(docs_train)
        # evaluate model
        spans_test = [get_annoation_tuples_from_doc(doc) for doc in docs_test]
        spans_pred = model.predict(docs_test)
        del model
        scores = calculate_spanF1(docs_test, spans_test, spans_pred, task_labels)
        calc_macro_averages(scores, verbose=False, overwrite_with_macro=True)
        scores_bin = calculate_binary_spanF1(spans_test, spans_pred, task_labels)
        scores.update(scores_bin)
        scores_df = pd.DataFrame({fname: [fval] for fname, fval in scores.items()})
        # log scores
        logger.info(f'Fold {fold_index+1} scores:')
        #logger.info("; ".join([f"{fname:4}: {fval:.3f}" for fname, fval in scores.items()]))
        # Log global F1, P, R first
        logger.info("; ".join([f"{metric}: {scores[metric]:.3f}" for metric in ['F1', 'P', 'R']]))
        for label in task_labels: # Then, log each label-specific triplet on its own line
            logger.info("; ".join([f"{label}-{metric}: {scores[f'{label}-{metric}']:.3f}" for metric in ['F1', 'P', 'R']]))
            logger.info("; ".join([f"{label}-{metric}-b: {scores[f'{label}-{metric}-b']:.3f}" for metric in ['F1', 'P', 'R']]))
        # formatted_values = [f"{col:10}: {scores[col].iloc[0]:.3f}" for col in scores.columns]
        results_df = pd.concat([results_df, scores_df], ignore_index=True)
        if pause_after_fold and fold_index < num_folds - 1:
            logger.info(f'Pausing for {pause_after_fold} minutes...')
            time.sleep(pause_after_fold * 60)
        rseed += 1; fold_index += 1
    logger.info('CROSSVALIDATION results:')
    for fname in scores.keys():
        logger.info(f'{fname:8}: ' + '; '.join(f'{nm}: {val:.3f}' for nm, val in results_df[fname].describe().items()))
    logger.info('Per-fold scores:')
    # for each score function, log all the per-fold results
    for fname in scores.keys():
        logger.info(f'{fname:8}: [{", ".join(f"{val:.3f}" for val in results_df[fname])}]')
    #print(results_df)

def build_seqlab_model(model_label: str, rseed: int, model_params: Dict, 
                       task_labels: List[str], task_indices: Dict[str, int]) -> OppSequenceLabelerMultitask:
    """
    Factory method to build a sequence labeling model.

    Args:
        model_label (str): Model label.
        rseed (int): Random seed for reproducibility.
        model_params (Dict): Dictionary of model parameters.
        task_labels (List[str]): List of task labels.
        task_indices (Dict[str, int]): Dictionary mapping task labels to task indices.

    Returns:
        OppSequenceLabelerMultitask: An instance of OppSequenceLabelerMultitask.
    """

    return OppSequenceLabelerMultitask(hf_model_label=model_label, rnd_seed=rseed,
                                       task_labels=task_labels, task_indices=task_indices,
                                       **model_params)

def run_seqlab_experiments(lang: str, num_folds: int, rnd_seed: int, test: Union[int, bool] = False, 
                           experim_label: str = None, pause_after_fold: int = 0, pause_after_model: int = 0, 
                           max_seq_length: int = 256, yml_config: dict = {}) -> None:
    """
    Run sequence labeling experiments.

    Args:
        lang (str): Language code.
        num_folds (int): Number of folds for cross-validation.
        rnd_seed (int): Random seed for reproducibility.
        test (Union[int, bool], optional): Number of train examples to use for testing, or False to use all data. Default is False.
        experim_label (str, optional): Experiment label. Default is None.
        pause_after_fold (int, optional): Pause duration in minutes after each fold. Default is 0.
        pause_after_model (int, optional): Pause duration in minutes after each model. Default is 0.
        max_seq_length (int, optional): Maximum sequence length. Default is 256.
        yml_config (dict, optional): Dictionary containing the configuration settings. Default is {}.

    Returns:
        None
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experim_label = f'{experim_label}_rseed_{rnd_seed}' if experim_label else f'rseed_{rnd_seed}'
    log_filename = f"seqlabel_experiments_{experim_label}_{timestamp}.log"
    setup_logging(log_filename)
    global logger
    logger = classif_experiment_runner.logger
    models = yml_config['model_list'][lang]
    params = copy(yml_config['hf_core_hparams'])
    params['lang'] = lang
    params['max_seq_length'] = max_seq_length
    logger.info(f'RUNNING classif. experiments: lang={lang.upper()}, num_folds={num_folds}, '
                f'max_seq_len={max_seq_length}, eval={params["eval"]}, rnd_seed={rnd_seed}, test={test}')
    logger.info(f'... SEED = {rnd_seed}')
    logger.info(f'... HPARAMS = { "; ".join(f"{param}: {val}" for param, val in yml_config["hf_core_hparams"].items())}')
    init_batch_size = params['batch_size']
    for model in models:
        try_batch_size = init_batch_size
        grad_accum_steps = 1
        while try_batch_size >= 1:
            try:
                params['batch_size'] = try_batch_size
                params['gradient_accumulation_steps'] = grad_accum_steps
                run_crossvalid_seqlab_transformers(lang=lang, model_label=model, model_params=params, num_folds=num_folds,
                                       rnd_seed=rnd_seed, test=test, pause_after_fold=pause_after_fold, src_langs=yml_config['src_langs'][lang])
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logging.warning(
                        f"GPU out of memory using batch size {try_batch_size}. Halving batch size and doubling gradient accumulation steps.")
                    try_batch_size //= 2
                    grad_accum_steps *= 2
                else:
                    raise e
            if try_batch_size < 1:
                logging.error("Minimum batch size reached and still encountering memory errors. Exiting.")
                break
        if pause_after_model:
            logger.info(f'Pausing for {pause_after_model} minutes...')
            time.sleep(pause_after_model * 60)

def spans_to_spanF1_format(ref_docs: List, spans: List[Tuple[str, int, int, str]]) -> Dict[str, List[List]]:
    """
    Convert a list of (label, start, end, author) tuples to the format used by the spanF1 scorer:
    map text_id: spans, where spans is a list of span lists, one per label; for each label spans
    are in the form [label, set of character indices].

    Args:
        ref_docs (List): List of reference documents.
        spans (List[Tuple[str, int, int, str]]): List of spans.

    Returns:
        Dict[str, List[List]]: Dictionary mapping text IDs to lists of spans.
    """

    result = {}
    for doc, span_list in zip(ref_docs, spans):
        text_id = get_doc_id(doc)
        if text_id not in result: result[text_id] = []
        labels = sorted(list(set([s[0] for s in span_list])))
        f1spans = []
        for l in labels:
            # take all spans with label l, and sort them by start index
            span_ranges = sorted([s[1:3] for s in span_list if s[0] == l], key=lambda x: x[0])
            # map each range to a set of character indices, using doc for offsets
            for start, end in span_ranges:
                first_char_index = doc[start].idx
                last_char_index = doc[end - 1].idx + len(doc[end - 1])
                f1spans.append([l, set(range(first_char_index, last_char_index))])
        result[text_id] = f1spans
    return result

def calculate_binary_spanF1(spans_test: List[List[Tuple[str, int, int, str]]], 
                            spans_predict: List[List[Tuple[str, int, int, str]]], task_labels: List[str]) -> Dict[str, float]:
    """
    Calculate binary classification metrics for occurrence vs. no occurrence of a given label in the text.
    Predictions of occurrence are derived from the spans predicted by the model.

    Args:
        spans_test (List[List[Tuple[str, int, int, str]]]): List of true spans.
        spans_predict (List[List[Tuple[str, int, int, str]]]): List of predicted spans.
        task_labels (List[str]): List of task labels.

    Returns:
        Dict[str, float]: Dictionary of binary classification metrics.
    """

    scoring_fns = classif_scores('span-binary')
    scores = {}
    for label in task_labels:
        # for each position int the spans_test and spans_predic
        # we have a list of spans, each span is a list of 4 elements: label, start, end, author
        # we need to extract binary 0-1 prediction for each position:
        # 0 it there is no span with label equal to the current label, 1 otherwise
        spans_test_bin = [1 if label in [s[0] for s in spans] else 0 for spans in spans_test]
        spans_predict_bin = [1 if label in [s[0] for s in spans] else 0 for spans in spans_predict]
        # Now we can calculate the binary metrics
        for metric, score_fn in scoring_fns.items():
            scores[f'{label}-{metric}-b'] = score_fn(spans_test_bin, spans_predict_bin)
    return scores

def calculate_spanF1(ref_docs: List, spans_test: List[List[Tuple[str, int, int, str]]], 
                     spans_predict: List[List[Tuple[str, int, int, str]]], task_labels: List[str], 
                     disable_logger: bool = True) -> Dict[str, float]:
    """
    Calculate span-level F1 scores for the predicted spans.

    Args:
        ref_docs (List): List of reference documents.
        spans_test (List[List[Tuple[str, int, int, str]]]): List of true spans.
        spans_predict (List[List[Tuple[str, int, int, str]]]): List of predicted spans.
        task_labels (List[str]): List of task labels.
        disable_logger (bool, optional): Whether to disable logging. Default is True.

    Returns:
        Dict[str, float]: Dictionary of span-level F1 scores.
    """

    spans_test_f1 = spans_to_spanF1_format(ref_docs, spans_test)
    spans_predict_f1 = spans_to_spanF1_format(ref_docs, spans_predict)
    return compute_score_pr(spans_predict_f1, spans_test_f1, task_labels, disable_logger=disable_logger)

def demo_experiment(lang: str, test_size: float = 0.2, num_epochs: int = 1, 
                    rnd_seed: int = 1443) -> None:
    """
    Run a demonstration experiment with a small dataset.

    Args:
        lang (str): Language code.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        num_epochs (int, optional): Number of training epochs. Default is 1.
        rnd_seed (int, optional): Random seed for reproducibility. Default is 1443.

    Returns:
        None
    """

    docs = load_dataset_full(lang, format='docbin')
    task_labels, task_indices = task_label_data()
    span_labels = [get_annoation_tuples_from_doc(doc) for doc in docs]
    docs_train, docs_test, spans_train, spans_test = \
        train_test_split(docs, span_labels, test_size=test_size, random_state=rnd_seed)
    seq_lab = OppSequenceLabelerMultitask(num_train_epochs=num_epochs, empty_label_ratio=0.1, hf_model_label='bert-base-cased',
                                          lang=lang, eval=None, rnd_seed=rnd_seed, task_labels=task_labels, task_indices=task_indices)
    seq_lab.fit(docs_train, spans_train)
    spans_predict = seq_lab.predict(docs_test)
    calculate_spanF1(docs_test, spans_test, spans_predict, task_labels=task_labels, disable_logger=False)

DEFAULT_RND_SEED = 1943123

def main() -> None:
    """
    Entry point function to accept command line arguments.

    Args:
        None

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Run Sequence Labelling Experiments")
    # Required arguments
    parser.add_argument("lang", type=str, help="Language")
    parser.add_argument("num_folds", type=int, help="Number of folds", default=5)
    parser.add_argument("--rnd_seed", type=int, help="Random seed", default=DEFAULT_RND_SEED)
    # Optional arguments
    parser.add_argument("--test", type=int, default=0, help="Number of train examples to use, for test, if 0 use all (no test)")
    parser.add_argument("--experim_label", type=str, default=None, help="Experiment label")
    parser.add_argument("--pause_after_fold", type=int, default=0, help="Pause duration after fold")
    parser.add_argument("--pause_after_model", type=int, default=0, help="Pause duration after model")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--gpu", type=int, default=0, help="index of the gpu for computation")
    parser.add_argument("--config_file", type=str, default=None, help="Path to a YAML configuration file")
    # Parse the arguments
    args = parser.parse_args()
    test = False if args.test == 0 else args.test
    print_cuda_devices()
    if args.config_file is None or not os.path.isfile(os.path.join(os.path.dirname(__file__), args.config_file)):
        raise ValueError("Please provide a valid configuration file.")
    yml_config = load_config_yml(os.path.join(os.path.dirname(__file__), args.config_file))
    # Call the function with parsed arguments
    run_seqlab_experiments(
        lang=args.lang,
        num_folds=args.num_folds,
        rnd_seed=args.rnd_seed,
        test=test,
        experim_label=args.experim_label,
        pause_after_fold=args.pause_after_fold,
        pause_after_model=args.pause_after_model,
        max_seq_length=args.max_seq_length,
        yml_config=yml_config
    )

def load_config_yml(
        config_file: str
    ) -> dict:
    """
    Load a YAML configuration file.
    
    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary containing the configuration settings.
    """

    import yaml
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    main()
    #demo_experiment('en', test_size=0.02, num_epochs=0.01)
