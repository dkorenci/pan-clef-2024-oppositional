"""
All functions that are used to process the dataset.

They all should accept this arguments or a subset of them:
    Tuple[pd.Series, pd.Series, pd.Series]: Texts, binary classes (1 - positive, 0 - negative), and text ids as pandas Series.

And return the same type of data.
"""

import pandas as pd
from typing import List

# process dataset text to mask specific words
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
