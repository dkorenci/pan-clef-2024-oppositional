import base64
from typing import Tuple, List

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
    positive_class = positive_class.lower()
    if positive_class == 'critical': return ['CONSPIRACY', 'CRITICAL']
    elif positive_class == 'conspiracy': return ['CRITICAL', 'CONSPIRACY']
    else: raise ValueError(f'Unknown positive class: {positive_class}')

def binary_labels_to_str(labels: List[int], positive_class: str) -> List[str]:
    '''
    Convert binary labels to string labels, using the positive class label.
    :param positive_class: 'conspiracy' or 'critical'
    :return: list of string labels
    '''
    positive_class = positive_class.lower()
    if positive_class == 'conspiracy': binmap = CATEGORY_MAPPING_CONSPIRACY_POS_INVERSE
    elif positive_class == 'critical': binmap = CATEGORY_MAPPING_CRITICAL_POS_INVERSE
    else: raise ValueError(f'Unknown positive class: {positive_class}')
    return [binmap[label] for label in labels]

def json_annotation_to_tuple(annot: dict) -> Tuple[str, int, int, str]:
    '''
    Convert a json-serialized annotation to a tuple of (label, start, end, author)
    :param annot:
    :return:
    '''
    return annot['category'], annot['start_spacy_token'], annot['end_spacy_token'], annot['annotator']

def tuple_to_json_annotation(annot: Tuple[str, int, int, str]) -> dict:
    '''
    Convert a tuple of (label, start, end, author) to a json-serializable annotation
    :param annot:
    :return:
    '''
    return {'category': annot[0], 'start_spacy_token': annot[1], 'end_spacy_token': annot[2], 'annotator': annot[3]}

def encode_spacy_tokens_as_bytes(spacy_tokens: List[str]) -> str:
    json_string = json.dumps(spacy_tokens)
    bytes_representation = base64.b64encode(json_string.encode('utf-8'))
    return bytes_representation.decode('utf-8')

def decode_spacy_tokens_from_bytes(encoded_string: str) -> List[str]:
    bytes_representation = base64.b64decode(encoded_string)
    spacy_tokens = json.loads(bytes_representation.decode('utf-8'))
    return spacy_tokens

def reconstruct_spacy_docs_from_json(json_file, lang, doc_categ_map=CATEGORY_MAPPING_CRITICAL_POS_INVERSE):
    '''
    Reconstruct Spacy Doc objects (for sequence labeling baseline) from a json file.
    The json records corresponding to texts should have the following keys: 'id' and 'spacy_tokens'
    If the records contain the 'category' and 'annotations' keys, this data will be added to the texts.
    :return:
    '''
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

def save_text_category_predictions_to_json(ids: List[str], predictions: List[str], json_file: str):
    data = [{'id': id, 'category': pred} for id, pred in zip(ids, predictions)]
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    with open(json_file, 'w', encoding='utf-8') as file:
        file.write(json_data)

def save_sequence_label_predictions_to_json(docs: List[Doc], preds: List[List[Tuple[str, int, int, str]]], json_file: str):
    '''
    :param list of spacy docs
    :param preds: predictions, for each document a list of tuples (label, token_start, token_end, author)
    :param json_file: filename to save the predictions to
    :return:
    '''
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


def span_annot_to_spanf1_format(annot: dict):
    '''
    Convert a span annotation to the format used for span-F1 score calculation.
    :param annot: dict with keys 'category', 'start_char', 'end_char'
    :return: list of [category, list of character indices]
    '''
    return [annot['category'], set(list(range(annot['start_char'], annot['end_char'])))]


def validate_json_annotations(annots: List):
    '''
    Validate the annotations of a single document, loaded from a .json file.
    :return:
    '''
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


def is_empty_annot(annots: List):
    '''
    Check if the list of annotations is empty, or contains only the NONE_LABEL annotation.
    :return:
    '''
    if len(annots) == 0: return True
    if len(annots) == 1 and annots[0]['category'] == NONE_LABEL: return True
    return False
