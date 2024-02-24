import logging, os
from copy import copy

from classif_experim.classif_experiment_runner import HF_CORE_HPARAMS, MAX_SEQ_LENGTH, build_transformer_model
from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.dataset_loaders import load_dataset_classification


def create_finetune_hparams(lang):
    ''' Create a set of hyperparameters reflecting that in classif_experiment_runner.py . '''
    params = copy(HF_CORE_HPARAMS)
    params['lang'] = lang
    params['eval'] = None
    params['max_seq_length'] = MAX_SEQ_LENGTH
    return params

def get_model_folder_name(lang, model_label, rseed, pos_cls):
    return f'classification_model_{model_label}_{lang}_rseed[{rseed}]_pos_class[{pos_cls}]'

def load_or_build_classif_fulltrain_model(lang, model_name, model_label, rseed=35412,
                                          positive_class='conspiracy'):
    mfolder = get_model_folder_name(lang, model_label, rseed, positive_class)
    current_module_folder = os.path.dirname(__file__)
    model_path = os.path.join(current_module_folder, mfolder)
    if os.path.exists(model_path):
        print('Model found, loading ...')
        return SklearnTransformerClassif.load(model_path)
    else:
        print(f'No model found at {mfolder}. Building model.')
        build_classif_model_on_full_train(lang, model_name, model_label, rseed=rseed, positive_class=positive_class, save=True)
        return SklearnTransformerClassif.load(mfolder)

def build_classif_model_on_full_train(lang, model_name, model_label, positive_class='conspiracy',
                                      rseed=35412, save=True):
    ''' Builds a finetuned transformer for conspiracy-critical classification. '''
    txt_tr, cls_tr, _ = load_dataset_classification(lang, string_labels=False, positive_class=positive_class)
    params = create_finetune_hparams(lang)
    #params['num_train_epochs'] = 0.5 # for testing
    try_batch_size = params['batch_size']
    mfolder = get_model_folder_name(lang, model_label, rseed, positive_class)
    grad_accum_steps = 1
    while try_batch_size >= 1:
        try:
            params['batch_size'] = try_batch_size
            params['gradient_accumulation_steps'] = grad_accum_steps
            model = build_transformer_model(model_name, params, rnd_seed=rseed)
            model.fit(txt_tr, cls_tr)
            if save: model.save(mfolder)
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
    build_classif_model_on_full_train('en', 'bert-base-cased', model_label='bert', test=True)
