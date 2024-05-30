'''
Example code for evaluation of the classification models on the official test dataset.
'''
from classif_experim.classif_model_builder import load_or_build_classif_fulltrain_model
from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.dataset_loaders import load_texts_and_ids_from_json
from data_tools.dataset_utils import binary_labels_to_str, save_text_category_predictions_to_json
from data_tools.evaluation_utils import run_official_evaluation_script
from settings import TEST_DATASET_EN, TEST_DATASET_ES

def evaluate_on_test_dataset(
        model: SklearnTransformerClassif, 
        lang: str, 
        positive_class: str
    ) -> None:
    """
    Applies the model to the test dataset and evaluates the predictions using the official evaluation script.
    
    Args:
        model (SklearnTransformerClassif): The classification model to be evaluated.
        lang (str): Language of the test dataset ('en' for English, 'es' for Spanish).
        positive_class (str): The positive class label ('conspiracy' or 'critical') used for prediction formatting.

    Returns:
        None
    """

    test_fname = TEST_DATASET_EN if lang == 'en' else TEST_DATASET_ES
    txt, ids = load_texts_and_ids_from_json(test_fname)
    cls_pred = model.predict(txt)
    cls_pred = binary_labels_to_str(cls_pred, positive_class)
    pred_fname = f'predictions_{lang}.json'
    save_text_category_predictions_to_json(ids, cls_pred, pred_fname)
    # run_official_evaluation_script('task1', pred_fname, test_fname)



def build_eval_model(
        lang: str, 
        hf_model_id: str,
        model_label: str,
        positive_class: str = 'conspiracy',
    ) -> None:
    """
    Builds or loads a full training classification model and evaluates it on the test dataset.
    
    Args:
        lang (str): Language of the model and dataset ('en' for English, 'es' for Spanish).
        positive_class (str, optional): The positive class label used for model training. Default is 'conspiracy'.

    Returns:
        None
    """
    model = load_or_build_classif_fulltrain_model(lang, hf_model_id, model_label=model_label,
                                              positive_class=positive_class)
    evaluate_on_test_dataset(model, lang, positive_class=positive_class)

if __name__ == '__main__':
    build_eval_model('en', 'jy46604790/Fake-News-Bert-Detect', 'fake-news-bert')
