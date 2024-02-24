import logging, os
from copy import copy

from classif_experim.hf_skelarn_wrapper import SklearnTransformerBase
from data_tools.dataset_loaders import load_dataset_full
from sequence_labeling.seqlabel_experiment_runner import HF_CORE_HPARAMS_SEQLAB_MULTITASK, build_seqlab_model, \
    task_label_data


def create_finetune_hparams_seqlab(lang):
    ''' Create a set of hyperparameters reflecting that in seqlabel_experiment_runner.py . '''
    params = copy(HF_CORE_HPARAMS_SEQLAB_MULTITASK)
    params['lang'] = lang
    params['eval'] = None
    params['max_seq_length'] = 256
    #params['num_train_epochs'] = 0.01 # quick training, for testing
    return params

def get_model_folder_name(lang, model_label, rseed):
    return f'seqlabel_model_{model_label}_{lang}_rseed[{rseed}]'

def load_or_build_seqlab_fulltrain_model(lang, hf_model_name, model_label, rseed=35412):
    mfolder = get_model_folder_name(lang, model_label, rseed)
    current_module_folder = os.path.dirname(__file__)
    model_path = os.path.join(current_module_folder, mfolder)
    if os.path.exists(model_path):
        print('Model found, loading ...')
        return SklearnTransformerBase.load(model_path)
    else:
        print(f'No model found at {mfolder}. Building model.')
        model = build_seqlab_model_on_full_train(lang, hf_model_name, model_label, rseed=rseed, save=False)
        return model

def build_seqlab_model_on_full_train(lang, hf_model_name, model_label, rseed=35412, save=False):
    ''' Builds a finetuned transformer for conspiracy-critical classification. '''
    docs = load_dataset_full(lang, format='docbin')
    params = create_finetune_hparams_seqlab(lang)
    try_batch_size = params['batch_size']
    mfolder = get_model_folder_name(lang, model_label, rseed)
    grad_accum_steps = 1
    task_labels, task_indices = task_label_data()
    while try_batch_size >= 1:
        try:
            params['batch_size'] = try_batch_size
            params['gradient_accumulation_steps'] = grad_accum_steps
            model = build_seqlab_model(hf_model_name, rseed=rseed, model_params=params,
                                       task_labels=task_labels, task_indices=task_indices)
            model.fit_(docs)
            if save:
                raise NotImplementedError("Saving the model is not yet implemented.")
                #TODO implement saving, modify SklearnTransformerBase save and load methods
                # to use MultiTaskModel save and load when appropriate
                # model.save(mfolder)
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
    build_seqlab_model_on_full_train('en', hf_model_name='bert-base-cased', model_label='bert')
