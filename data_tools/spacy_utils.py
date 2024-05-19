from typing import List, Tuple, Union
import spacy
from spacy.tokens import Doc

from data_tools.spacy_url_tokenization import SpacyURLTokenizer, customize_spacy_tokenizer

DEFAULT_ES_MODEL = 'es_core_news_sm'
DEFAULT_EN_MODEL = 'en_core_web_sm'

# names of the extension properties for spacy docs and spans
ON_DOC_EXTENSION = 'opn_spans' # doc._.opn_spans
#ON_SPAN_EXTENSION = 'label' # span._.label
#ON_SPAN_AUTHOR = 'author' # span._.author
ON_DOC_ID = 'doc_id' # doc._.id

# names of the extension property for doc class label
ON_DOC_CLS_EXTENSION = 'opn_class' # doc._.opn_spans

__EXTENSIONS_DEFINED = False

def define_spacy_extensions() -> None:
    """
    Define custom extensions for spaCy Doc objects.

    Args:
        None

    Returns:
        None
    """

    global __EXTENSIONS_DEFINED
    if __EXTENSIONS_DEFINED: return
    Doc.set_extension(ON_DOC_CLS_EXTENSION, default=None) # classif. label extension
    Doc.set_extension(ON_DOC_EXTENSION, default=[]) # span annotations
    Doc.set_extension(ON_DOC_ID, default=None) # doc id
    #Span.set_extension(ON_SPAN_EXTENSION, default=None)
    #Span.set_extension(ON_SPAN_AUTHOR, default=None)
    __EXTENSIONS_DEFINED = True

def print_doc_extensions(doc: Doc) -> None:
    """
    Print custom attributes of a spaCy Doc object.

    Args:
        doc (Doc): spaCy Doc object.

    Returns:
        None
    """

    # get all custom attributes from define_spacy_extensions() and print them
    for attr in dir(doc._):
        if attr.startswith('_'): continue
        print(f'{attr}: {getattr(doc._, attr)}')

def create_spacy_model(lang: str, fast: bool = False, url_tokenizer: bool = False) -> spacy.language.Language:
    """
    Create a spaCy language model with optional customizations.

    Args:
        lang (str): Language of the model ('en' for English, 'es' for Spanish).
        fast (bool, optional): If True, disable 'ner' and 'parser' components for faster loading. Default is False.
        url_tokenizer (bool, optional): If True, use the custom URL tokenizer. Default is False.

    Returns:
        spacy.language.Language: spaCy language model.
    """

    if lang == 'en': model = DEFAULT_EN_MODEL
    elif lang == 'es': model = DEFAULT_ES_MODEL
    else: raise ValueError(f'lang {lang} not supported')
    nlp = spacy.load(model) if not fast else spacy.load(model, disable=['ner', 'parser'])
    customize_spacy_tokenizer(nlp)
    if url_tokenizer: nlp.tokenizer = SpacyURLTokenizer(nlp)
    return nlp

def get_doc_id(doc: Doc) -> Union[str, None]:
    """
    Get the document ID of a spaCy Doc object.

    Args:
        doc (Doc): spaCy Doc object.

    Returns:
        Union[str, None]: Document ID.
    """

    return doc._.get(ON_DOC_ID)

def get_doc_class(doc: Doc) -> Union[str, None]:
    """
    Get the classification label of a spaCy Doc object.

    Args:
        doc (Doc): spaCy Doc object.

    Returns:
        Union[str, None]: Classification label.
    """

    return doc._.get(ON_DOC_CLS_EXTENSION)

def get_annoation_tuples_from_doc(doc: Doc) -> List[Tuple[str, int, int, str]]:
    """
    Helper to get annotation tuples from a spacy doc - just reads off the extension property
    that holds the list of annotations, and makes a copy.
    
    Args:
        doc (Doc): spaCy Doc object.

    Returns:
        List[Tuple[str, int, int, str]]: List of annotation tuples.
    """

    return list(doc._.get(ON_DOC_EXTENSION))
