'''
Example code for evaluation of the sequence labeling models on the official test dataset.
'''
from classif_experim.hf_skelarn_wrapper import SklearnTransformerBase
from data_tools.dataset_loaders import load_span_annotations_from_json
from data_tools.dataset_utils import reconstruct_spacy_docs_from_json, save_sequence_label_predictions_to_json
from data_tools.evaluation_utils import run_official_evaluation_script
from data_tools.spacy_utils import get_doc_id
from data_tools.span_data_definitions import SPAN_LABELS_OFFICIAL
from sequence_labeling.seqlabel_model_builder import load_or_build_seqlab_fulltrain_model
from sequence_labeling.span_f1_metric import compute_score_pr
from settings import TEST_DATASET_EN, TEST_DATASET_ES


def evaluate_seqlab_on_test_dataset(model: SklearnTransformerBase, lang: str):
    '''
    Applies the seq. label. model to the test dataset,
    and evaluates the predictions using the official evaluation script.
    '''
    test_fname = TEST_DATASET_EN if lang == 'en' else TEST_DATASET_ES
    test_docs = reconstruct_spacy_docs_from_json(test_fname, lang)
    spans_predict = model.predict(test_docs)
    pred_fname = f'seqlabel_predictions_{lang}.json'
    save_sequence_label_predictions_to_json(test_docs, spans_predict, pred_fname)
    run_official_evaluation_script('task2', pred_fname, test_fname)

def build_eval_seqlab_model(lang):
    hf_model_id = 'bert-base-cased' if lang == 'en' else 'dccuchile/bert-base-spanish-wwm-cased'
    model = load_or_build_seqlab_fulltrain_model(lang, hf_model_id, model_label='bert-baseline')
    evaluate_seqlab_on_test_dataset(model, lang)

if __name__ == '__main__':
    build_eval_seqlab_model('en')


