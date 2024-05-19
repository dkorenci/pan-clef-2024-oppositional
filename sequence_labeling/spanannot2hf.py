"""
Functionality for converting the span annotation from internal formats (spacy docs, lists of span tuples)
to huggingface format (datasets.Dataset objects).
"""
import random
from typing import List, Tuple

import numpy as np
import spacy
from spacy.tokens import Doc
from datasets import Dataset, Features, Sequence, ClassLabel, Value

from data_tools.spacy_utils import get_doc_id
from data_tools.span_data_definitions import NONE_LABEL, LABEL_DEF


def extract_spans(docs: List[Doc], spans_list: List[List[Tuple[str, int, int, str]]], 
                  downsample_empty: float = None, rnd_seed: int = 42, verbose: bool = False,
                  label_set: dict = LABEL_DEF, none_label: str = NONE_LABEL) -> dict:
    """
    Extract spans from spacy docs and span tuples.
    Return a map from label to list of (doc, span_indices_list) tuples.
    A document can have multiple spans of the same label, and can have multiple labels.
    If a label is not present for a document, it will be put in the map with an empty list,
    therefore a document will occur in the map for each label, not only the present ones.
    
    Args:
        docs (List[Doc]): List of spaCy Doc objects.
        spans_list (List[List[Tuple[str, int, int, str]]]): List of span annotations.
        downsample_empty (float, optional): Ratio for downsampling empty documents. Default is None.
        rnd_seed (int, optional): Random seed for downsampling. Default is 42.
        verbose (bool, optional): Flag for verbose output. Default is False.
        label_set (dict, optional): Set of labels to consider. Default is LABEL_DEF.
        none_label (str, optional): Label for empty spans. Default is NONE_LABEL.

    Returns:
        dict: Map from label to list of (doc, span_indices_list) tuples.
    """

    data = { l: [] for l in label_set }
    for doc, spans in zip(docs, spans_list):
        # local dictionary to group spans by label for the current doc
        # if a label is not present, its list will be empty
        label_to_spans = { l: [] for l in label_set }
        for label, start, end, _ in spans:
            if label != none_label:
                label_to_spans[label].append((start, end))
        for label, span_indices in label_to_spans.items():
            data[label].append((doc, span_indices))
    if downsample_empty is None: return data
    random.seed(rnd_seed)
    for label, entries in data.items():
        empty_entries = [entry for entry in entries if len(entry[1]) == 0]
        non_empty_entries = [entry for entry in entries if len(entry[1]) != 0]
        print(f'Empty label ratio for label {label}: '
              f'{len(empty_entries) / (len(empty_entries) + len(non_empty_entries)):.3f} : '
              f'{len(empty_entries)} out of {len(empty_entries) + len(non_empty_entries)}')
        num_empty_to_keep = int(len(non_empty_entries) * downsample_empty)
        if num_empty_to_keep < len(empty_entries):
            empty_entries = random.sample(empty_entries, num_empty_to_keep)
            if verbose: print(f'Label {label}: downsampling empty documents to {num_empty_to_keep} '
                              f'({len(non_empty_entries)} non-empty)')
        elif verbose:
            print(f'Label {label}: not enough empty documents, keeping all {len(empty_entries)} empty documents '
                  f'({len(non_empty_entries)} non-empty)')
        data[label] = empty_entries + non_empty_entries
    return data

def spanannot_tags_classlabel(task_label: str) -> ClassLabel:
    """
    Define hf Dataset ClassLabel for a single span label, constructs string tags,
    integer indices, and the mappings between them.
    
    Args:
        task_label (str): Task label.

    Returns:
        ClassLabel: Hugging Face ClassLabel object.
    """
    return ClassLabel(names=['O', f'B-{task_label}', f'I-{task_label}'])

def spanannot_feature_definition(label: str) -> Features:
    """
    Define the HF Dataset Features for a single span label.

    Args:
        label (str): Span label.

    Returns:
        Features: Hugging Face Features object.
    """

    Features({
        'tokens': Sequence(feature=Value(dtype='string'), length=-1),
        'ner_tags': Sequence(feature=spanannot_tags_classlabel(label), length=-1)
    })

def convert_to_hf_format(label: str, data: List[Tuple[Doc, List[Tuple[int, int]]]]) -> Dataset:
    """
    Create hf-compatible dataset from a list of (doc, span_indices_list) tuples, in BIO format,
    for a single label - tokens of each doc are assigned BIO tags according to the span indices.
    
    Args:
        label (str): Span label.
        data (List[Tuple[Doc, List[Tuple[int, int]]]]): List of (doc, span_indices_list) tuples.

    Returns:
        Dataset: Hugging Face Dataset object.
    """

    formatted_data = {
        'text_ids': [],
        'tokens': [],
        'ner_tags': []
    }
    tag_class = spanannot_tags_classlabel(label)
    for doc, span_indices_list in data:
        tokens = [token.text for token in doc]
        labels = ['O'] * len(tokens)
        # if a label is not present, its span_indices_list will be empty, so only 'O' labels will be present
        for start, end in span_indices_list:
            assert start < end, 'start must be smaller than end'
            labels[start] = 'B-' + label
            for idx in range(start + 1, end):
                labels[idx] = 'I-' + label
        formatted_data['tokens'].append(tokens)
        formatted_data['ner_tags'].append([tag_class.str2int(tag) for tag in labels])
        formatted_data['text_ids'].append(get_doc_id(doc))
    return Dataset.from_dict(formatted_data, features=spanannot_feature_definition(label))

def labels_from_predictions(preds: np.ndarray, tok_indices: List[int], task_label: str) -> List[str]:
    """
    For a specific seq. label task, convert the model's predictions to string BIO labels.
    
    Args:
        preds (np.ndarray): Matrix of logits or probabilities, shape (num_tokens, num_labels).
        tok_indices (List[int]): Hugging Face token indices.
        task_label (str): Task label.

    Returns:
        List[str]: List of BIO labels.
    """

    task_tags = spanannot_tags_classlabel(task_label) # definition of tasks's labels (string and int index)
    predictions = np.argmax(preds, axis=1) # predict the label with the highest probability
    labels = [task_tags.int2str(int(p)) for (p, l) in zip(predictions, tok_indices) if l != -100]
    return labels

def align_labels_with_tokens(tokens: List[str], labels: List[str]) -> List[str]:
    """
    If the orig. tokens are longer then the model's predicted labels,
    add 'O' labels to the end to make the lengths equal.
    
    Args:
        tokens (List[str]): List of tokens.
        labels (List[str]): List of labels.

    Returns:
        List[str]: Aligned labels.
    """

    N = len(tokens)
    assert len(labels) <= N
    labels = labels + ['O'] * (N - len(labels))
    return labels

def extract_span_ranges(tokens: List[str], bio_tags: List[str], allow_hanging_itag: bool = False) -> List[Tuple[int, int]]:
    """
    Extract the span ranges (token indices) from a list of tokens and BIO tags.
    
    Args:
        tokens (List[str]): List of tokens.
        bio_tags (List[str]): List of BIO tags.
        allow_hanging_itag (bool, optional): Flag to allow hanging 'I-' tags. Default is False.

    Returns:
        List[Tuple[int, int]]: List of span ranges.
    """

    span_ranges = []
    current_span = None
    for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
        if tag.startswith('B-'): # Start of a new entity
            if current_span is not None:
                span_ranges.append(current_span)
            current_span = (i, i + 1)
        elif tag.startswith('I-'): # Continuation of the current entity
            if current_span is not None:
                if current_span[1] != i:
                    span_ranges.append(current_span)
                current_span = (current_span[0], i + 1)
            else:
                if allow_hanging_itag: # Treat 'I-' tag as 'B-' if it's the first tag
                    current_span = (i, i + 1)
                else:
                    raise ValueError("Invalid BIO tagging sequence: 'I-' tag without preceding 'B-' tag.")
        else:
            if current_span is not None: # Outside of a span
                span_ranges.append(current_span)
            current_span = None
    if current_span is not None: # Add the last span if it exists
        span_ranges.append(current_span)
    return span_ranges

def test_extract_span_ranges():
    """
    Run test cases to validate the extraction of span ranges.
    """

    # Test case 1: Basic case with entities
    tokens1 = ["This", "is", "an", "example", "sentence", "about", "New", "York", "City", "."]
    bio_tags1 = ["B-L", "I-L", "I-L", "O", "O", "B-L", "I-L", "B-L", "I-L", "I-L"]
    spans1 = extract_span_ranges(tokens1, bio_tags1)
    assert spans1 == [(0, 3), (5, 7), (7, 10)]
    # Test case 2: Consecutive spans with no characters in between
    tokens2 = ["This", "is", "a", "test", "for", "BIO", "tagging", "."]
    bio_tags2 = ["B-L", "I-L", "I-L", "O", "B-L", "I-L", "I-L", "O"]
    spans2 = extract_span_ranges(tokens2, bio_tags2)
    assert spans2 == [(0, 3), (4, 7)]
    # Test case 3: Last tag not 'O'
    tokens3 = ["This", "is", "an", "example", "sentence", "."]
    bio_tags3 = ["O", "O", "O", "B-L", "I-L", "B-L"]
    spans3 = extract_span_ranges(tokens3, bio_tags3)
    assert spans3 == [(3, 5), (5, 6)]
    # Test case 4: 'I-' tag without preceding 'B-' tag
    tokens4 = ["Invalid", "B-L", "tagging", "sequence", "."]
    bio_tags4 = ["O", "I-L", "O", "O", "O"]
    try:
        spans4 = extract_span_ranges(tokens4, bio_tags4)
    except ValueError as e:
        assert str(e) == "Invalid BIO tagging sequence: 'I-' tag without preceding 'B-' tag."
    else:
        assert False, "Expected ValueError was not raised."
    # Test case 5: O-only tags with label 'L'
    tokens5 = ["This", "is", "a", "test", "."]
    bio_tags5 = ["O", "O", "O", "O", "O"]
    spans5 = extract_span_ranges(tokens5, bio_tags5)
    assert spans5 == []
    # CASES WITH HANGING 'I-' TAGS
    # Test case 1: 'I-' tag followed by 'B-' tag
    tokens1 = ["This", "is", "an", "example", "sentence", "about", "New", "York", "City", "."]
    bio_tags1 = ["I-L", "B-L", "I-L", "I-L", "O", "I-L", "B-L", "I-L", "I-L", "O"]
    spans1 = extract_span_ranges(tokens1, bio_tags1, allow_hanging_itag=True)
    assert spans1 == [(0, 1), (1, 4), (5, 6), (6, 9)]
    # Test case 2: Hanging 'I-' tag followed by 'B-' tag
    tokens2 = ["This", "is", "a", "test", "for", "BIO", "tagging", "."]
    bio_tags2 = ["I-L", "I-L", "I-L", "O", "B-L", "I-L", "I-L", "O"]
    spans2 = extract_span_ranges(tokens2, bio_tags2, allow_hanging_itag=True)
    assert spans2 == [(0, 3), (4, 7)]
    # Test case 3: Hanging 'I-' tag at the start followed by 'B-' tag
    tokens3 = ["This", "is", "an", "example", "sentence", "."]
    bio_tags3 = ["I-L", "B-L", "I-L", "I-L", "O", "I-L"]
    spans3 = extract_span_ranges(tokens3, bio_tags3, allow_hanging_itag=True)
    assert spans3 == [(0, 1), (1, 4), (5, 6)]
    # Test case 4: 'I-' tags only, allowed
    tokens4 = ["Test", "with", "only", "I-", "tags"]
    bio_tags4 = ["I-L", "I-L", "I-L", "I-L", "I-L"]
    spans4 = extract_span_ranges(tokens4, bio_tags4, allow_hanging_itag=True)
    assert spans4 == [(0, 5)]
    print("All test cases passed!")

def test_conversion():
    """
    Run test cases to validate the conversion of data to Hugging Face format.
    """

    nlp = spacy.load("en_core_web_sm")

    # Sample texts in English related to COVID vaccines, with mock annotations.
    texts = [
        "The goals of the vaccine are to prevent contagion and bolster immunity.",
        "The active agent in many vaccines is the mRNA.",
        "Facilitators have managed to distribute millions of doses throughout the country.",
        "Proponents of vaccination hope to achieve herd immunity soon.",
        "Victims of the virus hope to receive the vaccine as soon as possible.",
        "Negative effects of the vaccine are rare, but can include mild symptoms.",
        "This is a neutral sentence without any specific label."
    ]

    docs = [nlp(text) for text in texts]

    # Mock span annotations for the texts (using the short labels)
    spans_list = [
        [('O', 1, 4, 'author1')],
        [('A', 2, 7, 'author2')],
        [('F', 0, 2, 'author1')],
        [('P', 0, 3, 'author1')],
        [('V', 0, 3, 'author2')],
        [('E', 0, 4, 'author2')],
        []  # No label for the neutral sentence
    ]

    data_by_label = extract_spans(docs, spans_list)

    datasets = {}
    for label, data in data_by_label.items():
        datasets[label] = convert_to_hf_format(label, data)

    # Assertions to verify correctness
    assert datasets['O'][0]['ner_tags'][1:4] == ['B-O', 'I-O', 'I-O']
    assert datasets['A'][1]['ner_tags'][2:7] == ['B-A', 'I-A', 'I-A', 'I-A', 'I-A']
    assert datasets['F'][2]['ner_tags'][0:2] == ['B-F', 'I-F']
    assert datasets['P'][3]['ner_tags'][0:3] == ['B-P', 'I-P', 'I-P']
    assert datasets['V'][4]['ner_tags'][0:3] == ['B-V', 'I-V', 'I-V']
    assert datasets['E'][5]['ner_tags'][0:4] == ['B-E', 'I-E', 'I-E', 'I-E']
    print("All tests passed!")

if __name__ == "__main__":
    #test_conversion()
    test_extract_span_ranges()
