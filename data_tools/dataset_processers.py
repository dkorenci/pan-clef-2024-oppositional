"""
All functions that are used to process the dataset.

They all should accept this arguments or a subset of them:
    Tuple[pd.Series, pd.Series, pd.Series]: Texts, binary classes (1 - positive, 0 - negative), and text ids as pandas Series.

And return the same type of data.
"""

import copy
import json
import os
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, BatchEncoding

from data_tools.dataset_loaders import load_dataset_full
from data_tools.dataset_class import DatasetElement, dataset_to_dict, dataset_from_dict
from settings import TRAIN_DATASET_EN, TRAIN_DATASET_ES, TRAIN_TRANSLATED_DATASET_EN_ES, TRAIN_TRANSLATED_DATASET_ES_EN

def _load_translation_model(
        src_lang: str,
        dest_lang: str,
        device: str,
    ) -> Tuple[MarianMTModel, MarianTokenizer]:
    """
    Load a translation model for the specified source and destination languages.

    Args:
        src_lang (str): Source language code.
        dest_lang (str): Destination language code.
        device (str): Device to use for the model.

    Returns:
        Tuple[MarianMTModel, MarianTokenizer]: Translation model and tokenizer.
    """

    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return model, tokenizer

def _get_tokenized_chunks(text: str, tokenizer: MarianTokenizer, max_length: int) -> List[BatchEncoding]:
    """
    Split the input text into tokenized chunks of the specified maximum length.
    
    Args:
        text (str): Text to split.
        tokenizer (MarianTokenizer): Tokenizer for the translation model.
        max_length (int): Maximum length of the chunks.

    Returns:
        List[BatchEncoding]: List of tokenized chunks.
    """
    tokens = tokenizer(text, return_tensors='pt', truncation=False, verbose=False)

    tokenized_chunks = []

    # Because the tokenizer returns [1, n] tensors, we need to squeeze the first dimension
    input_ids = tokens['input_ids'].squeeze(0)
    attention_mask = tokens['attention_mask'].squeeze(0)

    for i in range(0, len(input_ids), max_length):
        chunk_input_ids = input_ids[i:i + max_length]
        chunk_attention_mask = attention_mask[i:i + max_length]

        # Same format as input: we need to unsqueeze the first dimension
        chunk = BatchEncoding({
            'input_ids': chunk_input_ids.unsqueeze(0),
            'attention_mask': chunk_attention_mask.unsqueeze(0)
        })
        tokenized_chunks.append(chunk)
    return tokenized_chunks

def _translate_text(text: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str:
    """
    Translate the input text using the specified translation model.

    Args:
        text (str): Text to translate.
        tokenizer (MarianTokenizer): Tokenizer for the translation model.
        model (MarianMTModel): Translation model.

    Returns
        str: Translated text.
    """
    max_length = model.config.max_length
    tokenized_chunks = _get_tokenized_chunks(text, tokenizer, max_length // 2)

    transladed_chunks = []
    for chunk in tokenized_chunks:
        chunk.to(model.device)
        chunk_outputs = model.generate(**chunk)
        translated_chunk = tokenizer.decode(chunk_outputs[0], skip_special_tokens=True)
        transladed_chunks.append(translated_chunk)
    return ' '.join(transladed_chunks)

def translate_dataset(
    dataset: List[DatasetElement],
    src_lang: str,
    dest_lang: str,
    device: str = 'cuda'
) -> List[DatasetElement]:
    """
    Translate the dataset to the specified destination language.

    Args:
        dataset (List[DatasetElement]): Dataset to translate.
        src_lang (str): Source language code.
        dest_lang (str): Destination language code.
        device (str, optional): Device to use for the model. Default is 'cuda'.
    
    Returns:
        List[DatasetElement]: Translated dataset.
    """
    translated_dataset = copy.deepcopy(dataset)
    model, tokenizer = _load_translation_model(src_lang, dest_lang, device)

    for element in tqdm(translated_dataset, desc="Translating texts"):
        element.text = _translate_text(element.text, tokenizer, model)
    return translated_dataset

def _correct_dataset_ids(
    dataset: List[DatasetElement],
) -> List[DatasetElement]:
    """
    Correct the text ids in the dataset to so there are not duplicates.

    Args:
        dataset (List[DatasetElement]): Original dataset.
        new_dataset (List[DatasetElement]): Translated dataset.
    
    Returns:
        List[DatasetElement]: Corrected translated dataset.
    """
    corected_dataset = copy.deepcopy(dataset)
    for element in corected_dataset:
        element.id = f'{element.id}_T'
    return corected_dataset

def get_translated_dataset(
    src_lang: str,
    dest_lang: str
) -> List[DatasetElement]:
    """
    Load the dataset in the source language, translate it to the destination language, and return the translated dataset.

    Args:
        src_lang (str): Source language code.
        dest_lang (str): Destination language code.
    
    Returns:
        List[DatasetElement]: Translated dataset.
    """
    if src_lang == 'en' and dest_lang == 'es':
        dataset_path = TRAIN_TRANSLATED_DATASET_EN_ES
    elif src_lang == 'es' and dest_lang == 'en':
        dataset_path = TRAIN_TRANSLATED_DATASET_ES_EN
    else:
        raise ValueError(f'Unknown language pair: {src_lang} -> {dest_lang}')

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    try:
        dataset = dataset_from_dict(load_dataset_full(dest_lang, format='json', translated='only'))
    except FileNotFoundError:
        dataset = dataset_from_dict(load_dataset_full(src_lang, format='json'))
        dataset = translate_dataset(dataset, src_lang, dest_lang)
        dataset = _correct_dataset_ids(dataset)

        dataset_dict = dataset_to_dict(dataset)
        with open(dataset_path, 'w', encoding='utf-8') as file:
            json.dump(dataset_dict, file, ensure_ascii=False, indent=4)
    return dataset

def mask_words(
        texts: pd.Series,
        mask_words: List[str],
        mask_token: str
        ) -> pd.Series:
    """
    Mask specific words in the dataset texts.
    
    Args:
        texts (pd.Series): Texts to be processed.
    Returns:
        pd.Series: Processed texts.
    """
    splitted_words = texts.apply(lambda text: text.split())
    masked_words = splitted_words.apply(lambda words: [mask_token if word in mask_words else word for word in words])
    masked_texts = masked_words.apply(lambda words: ' '.join(words))
    return masked_texts

if __name__ == '__main__':
    print('Translating dataset from English to Spanish')
    dataset = get_translated_dataset('en', 'es')
    print(f'Translated dataset size: {len(dataset)}')
    print(dataset[0])
    print(dataset[-1])
