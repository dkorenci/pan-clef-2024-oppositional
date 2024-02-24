'''
Example code for evaluation of the sequence labeling models on the official test dataset.
'''
from classif_experim.hf_skelarn_wrapper import SklearnTransformerBase
from data_tools.dataset_loaders import load_span_annotations_from_json
from data_tools.dataset_utils import reconstruct_spacy_docs_from_json, save_sequence_label_predictions_to_json
from data_tools.spacy_utils import get_doc_id
from data_tools.span_data_definitions import SPAN_LABELS_OFFICIAL
from sequence_labeling.seqlabel_model_builder import load_or_build_seqlab_fulltrain_model
from sequence_labeling.span_f1_metric import compute_score_pr
from settings import TEST_DATASET_EN, TEST_DATASET_ES


def evaluate_seqlab_on_test_dataset(model: SklearnTransformerBase, lang: str, test=None):
    '''
    :param positive_class: 'conspiracy' or 'critical', depending on how the model was trained
    :return:
    '''
    test_fname = TEST_DATASET_EN if lang == 'en' else TEST_DATASET_ES
    test_docs = reconstruct_spacy_docs_from_json(test_fname, lang)
    test_spans_f1 = load_span_annotations_from_json(test_fname, span_f1_format=True)
    if test: # use only #test documents
        test_docs = test_docs[:test]
        test_spans_f1 = {get_doc_id(d): test_spans_f1[get_doc_id(d)] for d in test_docs}
    spans_predict = model.predict(test_docs)
    save_sequence_label_predictions_to_json(test_docs, spans_predict, 'seqlabel_predictions.json')
    pred_spans_f1 = load_span_annotations_from_json('seqlabel_predictions.json', span_f1_format=True)
    assert set(test_spans_f1.keys()) == set(pred_spans_f1.keys()), "Document ids in the two annotation maps differ."
    all_span_labels = set(SPAN_LABELS_OFFICIAL.values())
    scores = compute_score_pr(pred_spans_f1, test_spans_f1, all_span_labels)
    print("; ".join([f"{metric}: {scores[metric]:.3f}" for metric in ['F1', 'P', 'R']]))
    for label in all_span_labels:  # Then, log each label-specific triplet on its own line
        print("; ".join([f"{label}-{metric}: {scores[f'{label}-{metric}']:.3f}" for metric in ['F1', 'P', 'R']]))

def build_eval_seqlab_model(lang):
    hf_model_id = 'bert-base-cased' if lang == 'en' else 'dccuchile/bert-base-spanish-wwm-cased'
    model = load_or_build_seqlab_fulltrain_model(lang, hf_model_id, model_label='bert-baseline')
    evaluate_seqlab_on_test_dataset(model, lang, test=None)

if __name__ == '__main__':
    build_eval_seqlab_model('en')


