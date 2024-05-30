import logging, os
from copy import copy

from classif_experim.classif_experiment_runner import MAX_SEQ_LENGTH, build_transformer_model, load_config_yml
from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.dataset_loaders import load_dataset_classification

HF_CORE_HPARAMS = {}

def create_finetune_hparams(
        lang: str
    ) -> dict:
    """
    Create a set of hyperparameters for fine-tuning based on the language.
    
    Args:
        lang (str): Language for the model ('en' for English, 'es' for Spanish).

    Returns:
        dict: Dictionary of hyperparameters.
    """
    yaml_config = load_config_yml(os.path.join(os.path.dirname(__file__), 'experiments/evaluate.yml'))

    # params = copy(HF_CORE_HPARAMS)
    params = copy(yaml_config['hf_core_hparams'])
    params['lang'] = lang
    params['eval'] = None
    params['max_seq_length'] = MAX_SEQ_LENGTH
    return params

def get_model_folder_name(
        lang: str, 
        model_label: str, 
        rseed: int, 
        pos_cls: str
    ) -> str:
    """
    Generate the folder name for saving the model based on its parameters.
    
    Args:
        lang (str): Language of the model ('en' for English, 'es' for Spanish).
        model_label (str): Label for the model.
        rseed (int): Random seed used for training.
        pos_cls (str): Positive class label used in training.

    Returns:
        str: Folder name for the model.
    """

    return f'classification_model_{model_label}_{lang}_rseed[{rseed}]_pos_class[{pos_cls}]'

def load_or_build_classif_fulltrain_model(
        lang: str, 
        model_name: str, 
        model_label: str, 
        rseed: int = 35412,
        positive_class: str = 'conspiracy'
    ) -> SklearnTransformerClassif:
    """
    Load an existing full training model or build a new one if not found.
    
    Args:
        lang (str): Language of the model ('en' for English, 'es' for Spanish).
        model_name (str): Identifier for the Hugging Face transformer model.
        model_label (str): Label for the model.
        rseed (int, optional): Random seed used for training. Default is 35412.
        positive_class (str, optional): Positive class label used in training. Default is 'conspiracy'.

    Returns:
        SklearnTransformerClassif: Loaded or newly built model.
    """

    mfolder = get_model_folder_name(lang, model_label, rseed, positive_class)
    current_module_folder = os.path.dirname(__file__)
    model_path = os.path.join(current_module_folder, mfolder)
    if os.path.exists(model_path):
        print('Model found, loading ...')
        return SklearnTransformerClassif.load(model_path)
    else:
        print(f'No model found at {mfolder}. Building model.')
        build_classif_model_on_full_train(lang, model_name, model_label, rseed=rseed, positive_class=positive_class, save=True)
        return SklearnTransformerClassif.load(model_path)

def build_classif_model_on_full_train(
        lang: str, 
        model_name: str, 
        model_label: str, 
        positive_class: str = 'conspiracy',
        rseed: int = 35412, 
        save: bool = True
    ) -> SklearnTransformerClassif:
    """
    Builds a fine-tuned transformer for conspiracy-critical classification.
    
    Args:
        lang (str): Language of the model ('en' for English, 'es' for Spanish).
        model_name (str): Identifier for the Hugging Face transformer model.
        model_label (str): Label for the model.
        positive_class (str, optional): Positive class label used in training. Default is 'conspiracy'.
        rseed (int, optional): Random seed used for training. Default is 35412.
        save (bool, optional): Whether to save the model after training. Default is True.

    Returns:
        SklearnTransformerClassif: Trained model.
    """

    txt_tr, cls_tr, _ = load_dataset_classification(lang, string_labels=False, positive_class=positive_class, src_langs=[[],[]])
    params = create_finetune_hparams(lang)
    #params['num_train_epochs'] = 0.5 # for testing
    try_batch_size = params['batch_size']
    mfolder = get_model_folder_name(lang, model_label, rseed, positive_class)
    model_path = os.path.join(os.path.dirname(__file__), mfolder)
    grad_accum_steps = 1
    while try_batch_size >= 1:
        try:
            params['batch_size'] = try_batch_size
            params['gradient_accumulation_steps'] = grad_accum_steps
            model = build_transformer_model(model_name, params, rnd_seed=rseed)
            model.fit(txt_tr, cls_tr)
            if save: model.save(model_path)
            return model
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
    return None

if __name__ == '__main__':
    build_classif_model_on_full_train('en', 'jy46604790/Fake-News-Bert-Detect', model_label='fake-news-bert')
