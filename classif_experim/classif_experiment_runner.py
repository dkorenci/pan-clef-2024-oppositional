from copy import copy
import logging
import time
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

from classif_experim.classif_utils import classif_scores
from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.dataset_loaders import load_dataset_classification
from data_tools.dataset_processers import mask_texts


def build_transformer_model(
        model_label: str, 
        model_hparams: dict,
        rnd_seed: int
    ) -> SklearnTransformerClassif:
    """
    Factory method for building a sklearn-wrapped transformer model.

    Args:
        model_label (str): Identifier for the Hugging Face transformer model.
        model_hparams (dict): Hyperparameters for the model training.
        rnd_seed (int): Random seed for reproducibility.

    Returns:
        SklearnTransformerClassif: An instance of the SklearnTransformerClassif model.
    """

    return SklearnTransformerClassif(hf_model_label=model_label, logger=logger, **model_hparams, rnd_seed=rnd_seed)

def get_tokenizer(model_label: str) -> str:
    """
    Get the tokenizer for a given Hugging Face transformer model.

    Args:
        model_label (str): Identifier for the Hugging Face transformer model.

    Returns:
        str: The tokenizer for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_label)
    return tokenizer


def run_classif_crossvalid(
        lang: str,
        src_langs: list,
        model_label: str, 
        model_params: dict,
        positive_class: str = 'critical', 
        default_mask_prob: float = 0.0,
        special_mask_prob: float = 0.0,
        num_folds: int = 5,
        rnd_seed: int = 3154561, 
        test: int = 0, 
        pause_after_fold: int = 0,
    ) -> dict:
    """
    Run k-fold cross-validation for a given model and report the results.

    Args:
        lang (str): Language of the dataset ('en' for English, 'es' for Spanish).
        src_langs (List[List[str]]): List of source languages for each target language.
        model_label (str): Identifier for the Hugging Face transformer model.
        model_params (dict): Hyperparameters for the model training.
        positive_class (str, optional): The positive class label used for model training. Default is 'critical'.
        mask (bool, optional): If true, mask specific words in the dataset. Default is False.
        num_folds (int, optional): Number of folds for cross-validation. Default is 5.
        rnd_seed (int, optional): Random seed for reproducibility. Default is 3154561.
        test (int, optional): If true, use a subset of the data for testing. Default is 0.
        pause_after_fold (int, optional): Minutes to pause after each fold. Default is 0.

    Returns:
        dict: Dictionary mapping text IDs to class predictions.
    """

    logger.info(f'RUNNING crossvalid. for model: {model_label}')
    score_fns = classif_scores('all')
    texts, classes, txt_ids = load_dataset_classification(lang, positive_class=positive_class, src_langs=src_langs)
    if test: texts, classes, txt_ids = texts[:test], classes[:test], txt_ids[:test]
    foldgen = StratifiedKFold(n_splits=num_folds, random_state=rnd_seed, shuffle=True)
    fold_index = 0
    results_df = pd.DataFrame(columns=score_fns.keys())
    conf_mx = None; rseed = rnd_seed
    pred_res = {} # map text_id -> class prediction
    for train_index, test_index in foldgen.split(texts, classes):
        logger.info(f'Starting Fold {fold_index+1}')
        model = build_transformer_model(model_label, model_params, rseed)
        logger.info(f'model built')
        # split data
        txt_tr, txt_tst = texts[train_index], texts[test_index]
        cls_tr, cls_tst = classes[train_index], classes[test_index]
        id_tst = txt_ids[test_index]
        if default_mask_prob > 0.0 or special_mask_prob > 0.0:
            mask_token = get_tokenizer(model_label).mask_token
            txt_tr = mask_texts(txt_tr, mask_token, lang, default_mask_prob, special_mask_prob)
        # train model
        model.fit(txt_tr, cls_tr)
        # evaluate model
        cls_pred = model.predict(txt_tst)
        for txt_id, pred in zip(id_tst, cls_pred):
            assert txt_id not in pred_res
            pred_res[txt_id] = pred
        del model # clear memory
        scores = pd.DataFrame({fname: [f(cls_tst, cls_pred)] for fname, f in score_fns.items()})
        # log scores
        logger.info(f'Fold {fold_index+1} scores:')
        logger.info("; ".join([f"{fname:10}: {f(cls_tst, cls_pred):.3f}" for fname, f in score_fns.items()]))
        # formatted_values = [f"{col:10}: {scores[col].iloc[0]:.3f}" for col in scores.columns]
        results_df = pd.concat([results_df, scores], ignore_index=True)
        conf_mx_tmp = confusion_matrix(cls_tst, cls_pred)
        if conf_mx is None: conf_mx = conf_mx_tmp
        else: conf_mx += conf_mx_tmp
        if pause_after_fold and fold_index < num_folds - 1:
            logger.info(f'Pausing for {pause_after_fold} minutes...')
            time.sleep(pause_after_fold * 60)
        rseed += 1; fold_index += 1
    conf_mx = conf_mx.astype('float64')
    conf_mx /= num_folds
    logger.info('CROSSVALIDATION results:')
    for fname in score_fns.keys():
        logger.info(f'{fname:10}: ' + '; '.join(f'{nm}: {val:.3f}' for nm, val in results_df[fname].describe().items()))
    logger.info('Per-fold scores:')
    # for each score function, log all the per-fold results
    for fname in score_fns.keys():
        logger.info(f'{fname:10}: [{", ".join(f"{val:.3f}" for val in results_df[fname])}]')
    logger.info('Confusion matrix:')
    for r in conf_mx:
        logger.info(', '.join(f'{v:7.2f}' for v in r))
    assert set(pred_res.keys()) == set(txt_ids)
    return pred_res

MAX_SEQ_LENGTH = 256

DEFAULT_RND_SEED = 564671

logger = None
def setup_logging(
        log_filename: str
    ) -> None:
    """
    Sets up logging to a file and console.
    
    Args:
        log_filename (str): Filename for the log file.

    Returns:
        None
    """

    global logger
    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above
        handlers=[
            logging.FileHandler(log_filename),  # Log to a file with timestamp in its name
            logging.StreamHandler()  # Log to console
        ]
    )
    logger = logging.getLogger('')

def run_classif_experiments(
        lang: str, 
        src_langs: list,
        num_folds: int, 
        rnd_seed: int, 
        test: bool = False, 
        experim_label: str = None,
        pause_after_fold: int = 0, 
        pause_after_model: int = 0, 
        max_seq_length: int = MAX_SEQ_LENGTH,
        positive_class: str = 'critical',  
        default_mask_prob: float = 0.0,
        special_mask_prob: float = 0.0,
        models: list = None,
        hf_core_hparams: dict = {},
    ) -> dict:
    """
    Run classification experiments, testing different models and configurations.
    
    Args:
        lang (str): Language of the dataset ('en' for English, 'es' for Spanish).
        src_langs (List[List[str]]): List of source languages for each target language.
        num_folds (int): Number of folds for cross-validation.
        rnd_seed (int): Random seed for reproducibility.
        test (bool, optional): If true, use a subset of the data for testing. Default is False.
        experim_label (str, optional): Label for the experiment. Default is None.
        pause_after_fold (int, optional): Minutes to pause after each fold. Default is 0.
        pause_after_model (int, optional): Minutes to pause after each model. Default is 0.
        max_seq_length (int, optional): Maximum sequence length for the model. Default is MAX_SEQ_LENGTH.
        positive_class (str, optional): The positive class label used for model training. Default is 'critical'.
        mask (bool, optional): If true, mask specific words in the dataset. Default is False.
        model_list (list, optional): List of model identifiers to test. Default is None.
        hf_core_hparams (dict, optional): Hyperparameters for the model training. Default is {}.

    Returns:
        dict: Dictionary mapping model identifiers to prediction results.
    """

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experim_label = f'{experim_label}_rseed_{rnd_seed}' if experim_label else f'rseed_{rnd_seed}'
    log_filename = f"cls_exp_{timestamp}_{experim_label}.log"
    setup_logging(log_filename)
    params = copy(hf_core_hparams)
    params['lang'] = lang
    params['max_seq_length'] = max_seq_length
    logger.info(f'RUNNING classif. experiments: lang={lang.upper()}, src_langs={src_langs}, num_folds={num_folds}, '
                f'default_mask_prob={default_mask_prob}, special_mask_prob={special_mask_prob}, positive_class={positive_class},'
                f'max_seq_len={max_seq_length}, eval={params["eval"]}, rnd_seed={rnd_seed}, test={test}')
    logger.info(f'... HPARAMS = {"; ".join(f"{param}: {val}" for param, val in hf_core_hparams.items())}')
    init_batch_size = params['batch_size']
    pred_res = {}
    for model in models:
        try_batch_size = init_batch_size
        grad_accum_steps = 1
        while try_batch_size >= 1:
            try:
                params['batch_size'] = try_batch_size
                params['gradient_accumulation_steps'] = grad_accum_steps
                res = run_classif_crossvalid(lang=lang, model_label=model, model_params=params, num_folds=num_folds,
                                             rnd_seed=rnd_seed, test=test, pause_after_fold=pause_after_fold,
                                             positive_class=positive_class, src_langs=src_langs,
                                             default_mask_prob=default_mask_prob, special_mask_prob=special_mask_prob)
                pred_res[model] = res
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
    return pred_res

def run_all_critic_conspi(
        seed: int = DEFAULT_RND_SEED, 
        langs: list = ['en', 'es'],
        num_folds: int = 5,
        test: bool = False,
        experim_label: str = None,
        pause_after_fold: int = 0,
        pause_after_model: int = 0,
        max_seq_length: int = MAX_SEQ_LENGTH,
        positive_class: str = 'critical',
        src_langs: list = [['en'], ['es']],
        default_mask_prob: float = 0.0,
        special_mask_prob: float = 0.0,
        model_list: dict = None,
        hf_core_hparams: dict = {}
    ) -> None:
    """
    Run classification experiments for multiple languages and classes.
    
    Args:
    seed (int, optional): Random seed for reproducibility. Default is DEFAULT_RND_SEED.
    langs (list, optional): List of languages to test. Default is ['en', 'es'].
    num_folds (int, optional): Number of folds for cross-validation. Default is 5.
    test (bool, optional): If true, use a subset of the data for testing. Default is False.
    experim_label (str, optional): Label for the experiment. Default is None.
    pause_after_fold (int, optional): Minutes to pause after each fold. Default is 0.
    pause_after_model (int, optional): Minutes to pause after each model. Default is 0.
    max_seq_length (int, optional): Maximum sequence length for the model. Default is MAX_SEQ_LENGTH.
    positive_class (str, optional): The positive class label used for model training. Default is 'critical'.
    src_langs (List[List[List[str]]], optional): List of source languages for each target language. Default is [['en'], ['es']].
    mask (bool, optional): If true, mask specific words in the dataset. Default is False.
    model_list (list, optional): List of model identifiers to test. Default is None.

    Returns:
    None
    """
    for index, lang in enumerate(langs):
        run_classif_experiments(lang=lang, num_folds=num_folds, rnd_seed=seed, test=test,
                    experim_label=experim_label, pause_after_fold=pause_after_fold,
                    pause_after_model=pause_after_model, max_seq_length=max_seq_length,
                    positive_class=positive_class, src_langs=src_langs[index], 
                    default_mask_prob=default_mask_prob, special_mask_prob=special_mask_prob,
                    models=model_list[lang], hf_core_hparams=hf_core_hparams)

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

def main(config_file: str) -> None:
    """
    Main function for running classification experiments.
    
    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        None
    """
    run_all_critic_conspi(**load_config_yml(config_file))

if __name__ == '__main__':
    import os
    main(os.path.join(os.path.dirname(__file__), 'experiments/ex4_final.yml'))
