import base64
from typing import Tuple, List, Union

import json
from spacy.tokens import Doc

from data_tools.spacy_utils import ON_DOC_ID, ON_DOC_CLS_EXTENSION, ON_DOC_EXTENSION, create_spacy_model, \
    define_spacy_extensions
from data_tools.span_data_definitions import SPAN_LABELS_OFFICIAL, NONE_LABEL

BINARY_MAPPING_CRITICAL_POS = {'CONSPIRACY': 0, 'CRITICAL': 1}
BINARY_MAPPING_CONSPIRACY_POS = {'CRITICAL': 0, 'CONSPIRACY': 1}

CATEGORY_MAPPING_CRITICAL_POS_INVERSE = {0: 'CONSPIRACY', 1: 'CRITICAL'}
CATEGORY_MAPPING_CONSPIRACY_POS_INVERSE = {0: 'CRITICAL', 1: 'CONSPIRACY'}

def classif_binary_str_labels(positive_class: str) -> List[str]:
    """
    Return a list of string labels for binary classification.

    Args:
        positive_class (str): The positive class label ('conspiracy' or 'critical').

    Returns:
        List[str]: List of string labels.
    """

    positive_class = positive_class.lower()
    if positive_class == 'critical': return ['CONSPIRACY', 'CRITICAL']
    elif positive_class == 'conspiracy': return ['CRITICAL', 'CONSPIRACY']
    else: raise ValueError(f'Unknown positive class: {positive_class}')

def binary_labels_to_str(labels: List[int], positive_class: str) -> List[str]:
    """
    Convert binary labels to string labels using the positive class label.

    Args:
        labels (List[int]): List of binary labels.
        positive_class (str): The positive class label ('conspiracy' or 'critical').

    Returns:
        List[str]: List of string labels.
    """

    positive_class = positive_class.lower()
    if positive_class == 'conspiracy': binmap = CATEGORY_MAPPING_CONSPIRACY_POS_INVERSE
    elif positive_class == 'critical': binmap = CATEGORY_MAPPING_CRITICAL_POS_INVERSE
    else: raise ValueError(f'Unknown positive class: {positive_class}')
    return [binmap[label] for label in labels]

def json_annotation_to_tuple(annot: dict) -> Tuple[str, int, int, str]:
    """
    Convert a json-serialized annotation to a tuple of (label, start, end, author).

    Args:
        annot (dict): Annotation in json format.

    Returns:
        Tuple[str, int, int, str]: Annotation as a tuple.
    """

    return annot['category'], annot['start_spacy_token'], annot['end_spacy_token'], annot['annotator']

def tuple_to_json_annotation(annot: Tuple[str, int, int, str]) -> dict:
    """
    Convert a tuple of (label, start, end, author) to a json-serializable annotation.

    Args:
        annot (Tuple[str, int, int, str]): Annotation as a tuple.

    Returns:
        dict: Annotation in json format.
    """

    return {'category': annot[0], 'start_spacy_token': annot[1], 'end_spacy_token': annot[2], 'annotator': annot[3]}

def encode_spacy_tokens_as_bytes(spacy_tokens: List[str]) -> str:
    """
    Encode spaCy tokens as a base64 string.

    Args:
        spacy_tokens (List[str]): List of spaCy tokens.

    Returns:
        str: Encoded tokens as a base64 string.
    """

    json_string = json.dumps(spacy_tokens)
    bytes_representation = base64.b64encode(json_string.encode('utf-8'))
    return bytes_representation.decode('utf-8')

def decode_spacy_tokens_from_bytes(encoded_string: str) -> List[str]:
    """
    Decode spaCy tokens from a base64 string.

    Args:
        encoded_string (str): Encoded tokens as a base64 string.

    Returns:
        List[str]: List of decoded spaCy tokens.
    """

    bytes_representation = base64.b64decode(encoded_string)
    spacy_tokens = json.loads(bytes_representation.decode('utf-8'))
    return spacy_tokens

def reconstruct_spacy_docs_from_json(json_file: str, lang: str, doc_categ_map: dict = CATEGORY_MAPPING_CRITICAL_POS_INVERSE) -> List[Doc]:
    """
    Reconstruct Spacy Doc objects (for sequence labeling baseline) from a json file.
    The json records corresponding to texts should have the following keys: 'id' and 'spacy_tokens'
    If the records contain the 'category' and 'annotations' keys, this data will be added to the texts.

    Args:
        json_file (str): Path to the json file.
        lang (str): Language of the documents.
        doc_categ_map (dict, optional): Mapping of document categories. Default is CATEGORY_MAPPING_CRITICAL_POS_INVERSE.

    Returns:
        List[Doc]: List of reconstructed spaCy Doc objects.
    """

    define_spacy_extensions()
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    nlp = create_spacy_model(lang, fast=True, url_tokenizer=True)
    # invert the category mapping, for reconstruction of spacy docs with binary labels
    doc_categ_map = {v: k for k, v in doc_categ_map.items()}
    recreated_docs = []
    for item in data:
        words = decode_spacy_tokens_from_bytes(item['spacy_tokens']) # use the spacy tokens from the json data
        #print(';'.join(words))
        doc = Doc(nlp.vocab, words=words) # Recreate the Doc object
        # set the doc id and category properties
        doc._.set(ON_DOC_ID, item['id'])
        if 'category' in item:
            doc._.set(ON_DOC_CLS_EXTENSION, doc_categ_map[item['category']])
        if 'annotations' in item:
            for annot in item.get('annotations', []): # add the annotations to doc (if any)
                # Calculate span from start/end token indices
                span_tuple = json_annotation_to_tuple(annot)
                doc._.get(ON_DOC_EXTENSION).append(span_tuple)
        recreated_docs.append(doc)
    return recreated_docs

def save_text_category_predictions_to_json(ids: List[str], predictions: List[str], json_file: str) -> None:
    """
    Save text category predictions to a json file.

    Args:
        ids (List[str]): List of text ids.
        predictions (List[str]): List of category predictions.
        json_file (str): Path to the json file.

    Returns:
        None
    """

    data = [{'id': id, 'category': pred} for id, pred in zip(ids, predictions)]
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    with open(json_file, 'w', encoding='utf-8') as file:
        file.write(json_data)

def save_sequence_label_predictions_to_json(docs: List[Doc], preds: List[List[Tuple[str, int, int, str]]], json_file: str) -> None:
    """
    Save sequence label predictions to a json file.

    Args:
        docs (List[Doc]): List of spaCy Doc objects.
        preds (List[List[Tuple[str, int, int, str]]]): Predictions for each document.
        json_file (str): Path to the json file.

    Returns:
        None
    """

    def convert_token_to_char_boundaries(doc, token_start, token_end):
        first_char_index = doc[token_start].idx
        last_char_index = doc[token_end - 1].idx + len(doc[token_end - 1])
        return first_char_index, last_char_index
    json_docs = []
    for doc, seq_annots in zip(docs, preds):
        doc_id = doc._.get(ON_DOC_ID)
        doc_json = {'id': doc_id, 'annotations': []}
        for annot in seq_annots:
            label, token_start, token_end, author = annot
            start_char, end_char = convert_token_to_char_boundaries(doc, token_start, token_end)
            doc_json['annotations'].append({'category': label, 'start_char': start_char, 'end_char': end_char, 'annotator': author})
        json_docs.append(doc_json)
    # save to file
    json_data = json.dumps(json_docs, ensure_ascii=False, indent=2)
    with open(json_file, 'w', encoding='utf-8') as file:
        file.write(json_data)

if __name__ == '__main__':
    pass

def span_annot_to_spanf1_format(annot: dict) -> List[Union[str, set]]:
    """
    Convert a span annotation to the format used for span-F1 score calculation.

    Args:
        annot (dict): Span annotation in json format.

    Returns:
        List[Union[str, set]]: Span annotation in span-F1 format.
    """

    return [annot['category'], set(list(range(annot['start_char'], annot['end_char'])))]

def validate_json_annotations(annots: List[dict]) -> None:
    """
    Validate the annotations of a single document loaded from a json file.

    Args:
        annots (List[dict]): List of annotations.

    Returns:
        None
    """

    if len(annots) == 0: return # can be an empty list
    valid_labels = set(SPAN_LABELS_OFFICIAL.values())
    for annot in annots:
        if annot['category'] == NONE_LABEL:
            if len(annots) > 1:
                raise ValueError(f'Set of document annotations can only contain '
                                 f'a single {NONE_LABEL} annotation. Found:\n{annots}')
        else:
            # assert that 'category', 'start_char', 'end_char' are keys in the annotation
            if not all([key in annot for key in ['category', 'start_char', 'end_char']]):
                raise ValueError(f'Invalid annotation missing required keys: {annot}')
            # assert that 'category' is in the set of valid span categories
            if annot['category'] not in valid_labels:
                raise ValueError(f'Invalid annotation category: {annot}')

def is_empty_annot(annots: List[dict]) -> bool:
    """
    Check if the list of annotations is empty or contains only the NONE_LABEL annotation.

    Args:
        annots (List[dict]): List of annotations.

    Returns:
        bool: True if the list is empty or contains only NONE_LABEL, False otherwise.
    """

    if len(annots) == 0: return True
    if len(annots) == 1 and annots[0]['category'] == NONE_LABEL: return True
    return False
