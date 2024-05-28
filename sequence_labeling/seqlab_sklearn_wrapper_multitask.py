from logging import Logger
import random
from typing import List, Tuple, Dict, Union

import datasets
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc
from torch import tensor as TT
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainerCallback, EarlyStoppingCallback

from classif_experim.hf_skelarn_wrapper import SklearnTransformerBase
from data_tools.spacy_utils import get_doc_id, get_annoation_tuples_from_doc
from sequence_labeling.multi_task_model import MultiTaskModel, Task
from sequence_labeling.spanannot2hf import extract_spans, convert_to_hf_format, labels_from_predictions, \
    align_labels_with_tokens, extract_span_ranges

class LoggingCallback(TrainerCallback):
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info(f'Epoch {state.epoch} ended')

class OppSequenceLabelerMultitask(SklearnTransformerBase):
    """
    'Oppositional Sequence Labeler', wraps the data transformation functionality and the multitask HF model
    for sequence labeling into a sklearn-like interface.
    """

    def __init__(self, task_labels: List[str], task_indices: Dict[str, int], empty_label_ratio: float = 0.2, loss_freq_weights: bool = False, task_importance: Dict[str, float] = None, **kwargs) -> None:
        """
        Initialize the OppSequenceLabelerMultitask.

        Args:
            task_labels (List[str]): List of task labels.
            task_indices (Dict[str, int]): Dictionary mapping task labels to task indices.
            empty_label_ratio (float, optional): Ratio of empty labels. Default is 0.2.
            loss_freq_weights (bool, optional): Whether to use frequency-based loss weights. Default is False.
            task_importance (Dict[str, float], optional): Dictionary of task importance weights. Default is None.
            **kwargs: Additional arguments for the superclass.

        Returns:
            None
        """

        super().__init__(**kwargs)
        self._empty_label_ratio = empty_label_ratio
        self._loss_freq_weights = loss_freq_weights
        self._task_importance = task_importance
        self.task_labels, self.task_indices = task_labels, task_indices

    def dataset_stats(self, docs: List[Doc]) -> None:
        """
        Calculate and print statistics of the transformed dataset that will be used for training.

        Args:
            docs (List[Doc]): List of SpaCy documents.

        Returns:
            None
        """

        old_eval = self._eval
        span_labels = [get_annoation_tuples_from_doc(doc) for doc in docs]
        self._eval = None # entire dataset will be the train set
        self._init_tokenizer()
        #self._init_model()
        #self._init_temp_folder()
        self._construct_train_eval_raw_datasets(docs, span_labels)
        self._calc_dset_stats(verbose=True)
        #self._do_training()
        #self._cleanup_temp_folder()
        self._eval = old_eval

    def _calc_dset_stats(self, verbose: bool = False) -> None:
        """
        Calculate and print dataset statistics using self._raw_train.

        Args:
            verbose (bool, optional): Whether to print detailed statistics. Default is False.

        Returns:
            None
        """

        if verbose: print(f'Seqlabel train dataset statistics for {self._lang}:')
        # use self._raw_train
        # for each label, count and print the number of instances, as well as the number of instances with no spans
        # print these statistics, use fixed ordering of the labels
        num = sum([len(self._raw_train[label]) for label in self.task_labels])
        if verbose: print(f'Total: {num} instances')
        self._task_frequencies = {}
        for label in self.task_labels:
            dset = self._raw_train[label]
            num_instances = len(dset)
            num_empty = len([1 for i in range(num_instances) if list(set(dset[i]['ner_tags'])) == [0]])
            self._task_frequencies[label] = num_instances/num
            if verbose:
                print(f'{label:<3}: {num_instances:<5} instances ({num_instances/num*100:5.3f}%) '
                    f'{num_empty:<5} empty ({num_empty/num*100:5.3f}%)')

    def fit(self, docs: List[Doc], span_labels: List[Tuple[str, int, int, str]]) -> None:
        """
        Fit the model on the provided dataset.

        Args:
            docs (List[Doc]): List of SpaCy documents.
            span_labels (List[Tuple[str, int, int, str]]): List of lists of spans, each span is a tuple of (label, start, end, text).

        Returns:
            None
        """

        self._init_tokenizer()
        self._construct_datasets_for_inference(docs, span_labels)
        self._init_model()
        self._init_temp_folder()
        self._do_training()
        self._cleanup_temp_folder()
        # input txt formatting and tokenization
        # training

    def fit_(self, docs: List[Doc]) -> None:
        """
        Helper method to enable fitting without previously extracting spans into a separate list.

        Args:
            docs (List[Doc]): List of SpaCy documents with annotated spans.

        Returns:
            None
        """

        span_labels = [get_annoation_tuples_from_doc(doc) for doc in docs]
        self.fit(docs, span_labels)

    def _do_training(self) -> None:
        """
        Perform the training process for the model.

        Args:
            None

        Returns:
            None
        """

        train_dataset = self._dataset['train']
        eval_dataset = self._dataset['eval'] if self._eval else None
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self._init_train_args()
        trainer = Trainer(model=self.model, args=self._training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset if self._eval else None,
            tokenizer=self.tokenizer, data_collator=data_collator,
            callbacks=[
                # EarlyStoppingCallback(
                #     early_stopping_patience=self._stop_patience,
                #     early_stopping_threshold=self._rel_stop_threshold
                # ) if self._eval_metric is not None else None,
                # LoggingCallback(self._logger) if self._logger is not None else None,
            ],
        )
        trainer.train()
        if self.model is not trainer.model: # just in case
            del self.model
            self.model = trainer.model
        del trainer
        torch.cuda.empty_cache()

    def _construct_datasets_for_inference(self, docs: List[Doc], spans: List[List[Tuple[str, int, int, str]]] = None) -> None:
        """
        Construct datasets for inference.

        Args:
            docs (List[Doc]): List of SpaCy documents.
            spans (List[List[Tuple[str, int, int, str]]], optional): List of spans for each document. Default is None.

        Returns:
            None
        """

        self._construct_train_eval_raw_datasets(docs, spans)
        self._calc_dset_stats(verbose=False)
        #self._inspect_data(self._raw_train, self.span_labels, num_samples=5)
        self._hf_tokenize_task_dataset()

    def _construct_train_eval_raw_datasets(self, docs: List[Doc], spans: List[List[Tuple[str, int, int, str]]]) -> None:
        """
        Construct raw datasets for training and evaluation.

        Args:
            docs (List[Doc]): List of SpaCy documents.
            spans (List[List[Tuple[str, int, int, str]]]): List of spans for each document.

        Returns:
            None
        """

        if self._eval:
            docs_train, docs_eval, spans_train, spans_eval = \
                train_test_split(docs, spans, test_size=self._eval, random_state=self._rnd_seed)
            self._raw_train = self._construct_raw_hf_dataset(docs_train, spans_train, downsample=self._empty_label_ratio)
            self._raw_eval = self._construct_raw_hf_dataset(docs_eval, spans_eval, downsample=self._empty_label_ratio)
        else:
            self._raw_train = self._construct_raw_hf_dataset(docs, spans, downsample=self._empty_label_ratio)
            self._raw_eval = None

    def _construct_raw_hf_dataset(self, docs: List[Doc], span_labels: List[List[Tuple[str, int, int, str]]], downsample: float) -> Dict[str, Dataset]:
        """
        Convert the data to Hugging Face format: create one HF dataset per label in BIO format.

        Args:
            docs (List[Doc]): List of SpaCy documents.
            span_labels (List[List[Tuple[str, int, int, str]]]): List of spans for each document.
            downsample (float): Downsample ratio for empty labels.

        Returns:
            Dict[str, Dataset]: Dictionary of HF datasets per label.
        """

        data_by_label = extract_spans(docs, span_labels, downsample_empty=downsample, rnd_seed=self._rnd_seed,
                                      label_set=self.task_labels)
        datasets = {}
        for label, data in data_by_label.items():
            datasets[label] = convert_to_hf_format(label, data)
        return datasets

    def _init_tokenizer(self) -> None:
        """
        Initialize the tokenizer.

        Args:
            None

        Returns:
            None
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_label)
        if isinstance(self.tokenizer, transformers.RobertaTokenizerFast):
            self.tokenizer.add_prefix_space = True
        self.tokenizer_params = {'truncation': True}
        if self._max_seq_length is not None: self.tokenizer_params['max_length'] = self._max_seq_length

    def _get_narrative_tasks(self) -> List[Task]:
        """
        Definition of tasks for this sequence labeling problem, compatible with MultiTaskModel.

        Args:
            None

        Returns:
            List[Task]: List of tasks for the model.
        """

        return [
            Task(id=self.task_indices[task_id], name=None, num_labels=3, type="token_classification")
            for task_id in self.task_labels
        ]

    def _calculate_task_weights(self) -> None:
        """
        Calculate task weights from task frequencies and importance weights.

        Args:
            None

        Returns:
            None
        """

        def normalize_weight_map(weights):
            sum_weights = sum(weights.values())
            for label in self.task_labels: weights[self.task_indices[label]] /= sum_weights
        if not self._loss_freq_weights and not self._task_importance:
            self._task_weights = None
        else: # calculate freq. weights, importance weights, or their combination if both are provided
            self._task_weights = {}
            if self._loss_freq_weights: # use loss frequency weights
                for label in self.task_labels: self._task_weights[self.task_indices[label]] = 1/self._task_frequencies[label]
                normalize_weight_map(self._task_weights)
                if self._task_importance:
                    for label in self.task_labels: self._task_weights[self.task_indices[label]] *= self._task_importance[label]
                    normalize_weight_map(self._task_weights)
            else:
                for label in self.task_labels: self._task_weights[self.task_indices[label]] = self._task_importance[label]
                normalize_weight_map(self._task_weights)

    def _init_model(self) -> None:
        """
        Initialize the model.

        Args:
            None

        Returns:
            None
        """

        self._calculate_task_weights()
        self._is_roberta = 'roberta' in self._hf_model_label.lower()
        self.model = MultiTaskModel(self._hf_model_label, self._get_narrative_tasks(), task_weights=self._task_weights).to(self._device)

    def _hf_tokenize_task_dataset(self) -> None:
        """
        Given a HF dataset for a single task (label) produced by _construct_hf_datasets,
        tokenize it, add taks labels, and return the tokenized dataset.
        
        Args:
            None

        Returns:
            None
        """

        # for each label, perform hf tokenization
        raw_dset_per_label = {}
        for label in self.task_labels:
            if not self._eval: raw_dataset = DatasetDict({'train': self._raw_train[label]})
            else: raw_dataset = DatasetDict({'train': self._raw_train[label], 'eval': self._raw_eval[label]})
            #label_list = raw_dataset['train'].features['ner_tags'].feature.names
            tokenized_dataset = self._tokenize_token_classification_dataset(
                raw_datasets=raw_dataset, tokenizer=self.tokenizer, task_id=self.task_indices[label])
            raw_dset_per_label[label] = tokenized_dataset
        # merge per-label datasets into one, for multi-task training
        dset_splits = ['train', 'eval'] if self._eval else ['train']
        datasets_df = { split: None for split in dset_splits }
        for label, raw_dset in raw_dset_per_label.items(): # merge datasets as pandas dataframes
            for split in dset_splits:
                if datasets_df[split] is None:
                    datasets_df[split] = raw_dset[split].to_pandas()
                else:
                    datasets_df[split] = pd.concat([datasets_df[split], raw_dset[split].to_pandas()], ignore_index=True)
        # convert dataframes backt to HF datasets, shuffle, and create final dataset dict
        merged_datasets = { split: datasets.Dataset.from_pandas(datasets_df[split]) for split in dset_splits }
        for split in dset_splits:
            merged_datasets[split].shuffle(seed=self._rnd_seed)
        if self._eval is None:
            self._dataset = datasets.DatasetDict({'train': merged_datasets['train']})
        else:
            self._dataset = datasets.DatasetDict({'train': merged_datasets['train'], 'eval': merged_datasets['eval']})

    def _tokenize_token_classification_dataset(self, raw_datasets: Union[DatasetDict, Dataset], 
                                               tokenizer: AutoTokenizer, task_id: int, text_column_name: str = 'tokens', 
                                               label_column_name: str = 'ner_tags') -> DatasetDict:
        """
        Tokenize the token classification dataset.

        Args:
            raw_datasets (Union[DatasetDict, Dataset]): Raw HF datasets.
            tokenizer (AutoTokenizer): Tokenizer to use.
            task_id (int): Task ID.
            text_column_name (str, optional): Name of the text column. Default is 'tokens'.
            label_column_name (str, optional): Name of the label column. Default is 'ner_tags'.

        Returns:
            DatasetDict: Tokenized HF datasets.
        """

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=True,
                truncation=True,
                max_length=self._max_seq_length,
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None: label_ids.append(-100) # for special token set label to -100 (to ignore in loss)
                    elif word_idx != previous_word_idx: # set the label only for the first token of a word
                        label_ids.append(label[word_idx])
                    else: label_ids.append(-100) # for consecutive tokens of multi-token words, set the label to -100
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
            return tokenized_inputs
        if isinstance(raw_datasets, DatasetDict):
            tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True, num_proc=1,
                                load_from_cache_file=False)
        elif isinstance(raw_datasets, Dataset): # helper code that enables to tokenize a single Dataset as DatasetDict
            dset = DatasetDict({'dset': raw_datasets})
            tokenized_datasets = dset.map(tokenize_and_align_labels, batched=True, num_proc=1,
                                load_from_cache_file=False)
            tokenized_datasets = tokenized_datasets['dset']
        else: raise ValueError(f'Unknown dataset type: {type(raw_datasets)}')
        return tokenized_datasets

    def _inspect_data(self, datasets: Dict[str, Dataset], label_list: List[str], num_samples: int = 10) -> None:
        """
        Inspect and print sample data for each label.

        Args:
            datasets (Dict[str, Dataset]): Dictionary of datasets per label.
            label_list (List[str]): List of task labels.
            num_samples (int, optional): Number of samples to inspect. Default is 10.

        Returns:
            None
        """

        def print_aligned_tokens_and_tags(tokens, ner_tags):
            token_str = ' '.join([f"{token:<{len(token) + 2}}" for token in tokens])
            print(token_str)
            ner_tag_str = ' '.join([f"{str(tag):<{len(tokens[i]) + 2}}" for i, tag in enumerate(ner_tags)])
            print(ner_tag_str)
            print("\n" + "-" * 40)

        for label in label_list:
            dataset = datasets[label]
            sampled_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            print(f"===== ORIGINAL TEXT FOR LABEL: {label} =====")
            for idx in sampled_indices:
                tokens = dataset[idx]['tokens']
                ner_tags = dataset[idx]['ner_tags']
                print_aligned_tokens_and_tags(tokens, ner_tags)
            tokenized_dataset = \
            self._tokenize_token_classification_dataset(DatasetDict({'train': dataset}),
                                                        self.tokenizer, self.task_indices[label])['train']
            print(f"\n===== TOKENIZED TEXT FOR LABEL: {label} =====")
            for idx in sampled_indices:
                tokens = self.tokenizer.convert_ids_to_tokens(tokenized_dataset[idx]['input_ids'])
                labels = tokenized_dataset[idx]['labels']
                print_aligned_tokens_and_tags(tokens, labels)
            print("\n\n")

    def _construct_predict_dataset(self, docs: List[Doc], spans: List[List[Tuple[str, int, int, str]]] = None) -> Dict[str, Dataset]:
        """
        Construct the dataset for prediction.

        Args:
            docs (List[Doc]): List of SpaCy documents.
            spans (List[List[Tuple[str, int, int, str]]], optional): List of spans for each document. Default is None.

        Returns:
            Dict[str, Dataset]: Dictionary of HF datasets per label for prediction.
        """

        if spans == None: # no spans provided, create a list of #docs empty lists for compatibility
            spans = [[] for _ in range(len(docs))]
        raw_dset_per_label = self._construct_raw_hf_dataset(docs, spans, downsample=None)
        tokenized_dset_per_label = {}
        for label in self.task_labels:
            tokenized_dataset = self._tokenize_token_classification_dataset(
                raw_datasets=raw_dset_per_label[label], tokenizer=self.tokenizer, task_id=self.task_indices[label])
            tokenized_dset_per_label[label] = tokenized_dataset
        return tokenized_dset_per_label

    def predict(self, X: List[Doc]) -> List[List[Tuple[str, int, int, str]]]:
        """
        Predict the spans for the given documents.

        Args:
            X (List[Doc]): List of SpaCy documents.

        Returns:
            List[List[Tuple[str, int, int, str]]]: List of lists of spans (full annotations for one document); 
            each span is a tuple of (label, start, end, author), for the data to be in the same format as in the original spacy data
        """

        text_labels_pred = {} # intermediate map with output, and the helper function for adding labels to it
        def add_labels_to_map(label_map, text_id, task_label, labels: List[str]):
            if text_id not in label_map: label_map[text_id] = {}
            if task_label not in label_map[text_id]: label_map[text_id][task_label] = labels
            else: raise ValueError(f'Label map already contains labels for text id {text_id} and task {task_label}')
        # tokenize input, predict, transform data
        tokenized_dset_per_label = self._construct_predict_dataset(X)
        for label in self.task_labels:
            dset = tokenized_dset_per_label[label]
            for t in dset:
                if not self._is_roberta: ttids = t['token_type_ids']
                else: ttids = [0]
                ids, att, tti, tsk = [t['input_ids']], [t['attention_mask']], [ttids], [self.task_indices[label]]
                ids, att, tti, tsk = TT(ids, device=self.device), TT(att, device=self.device), \
                                     TT(tti, device=self.device), TT(tsk, device=self.device)
                res, _ = self.model(ids, att, tti, task_ids=tsk)
                preds = res[0].cpu().detach().numpy()
                orig_tokens = t['tokens']
                pred_labels = labels_from_predictions(preds, t['labels'], label)
                pred_labels = align_labels_with_tokens(orig_tokens, pred_labels)
                add_labels_to_map(text_labels_pred, t['text_ids'], label, pred_labels)
        # convert the map to the format of the original spacy data
        id2doc = {get_doc_id(doc): doc for doc in X}
        span_labels_pred = {text_id:[] for text_id in text_labels_pred.keys()}
        for text_id in text_labels_pred.keys():
            for task_label in text_labels_pred[text_id].keys():
                span_bio_tags = text_labels_pred[text_id][task_label]
                if len(span_bio_tags) == 0: continue
                doc = id2doc[text_id]
                # assert that doc has the same number of tokens as there are bio tags
                assert len(doc) == len(span_bio_tags)
                tokens = [token.text for token in doc]
                spans = extract_span_ranges(tokens, span_bio_tags, allow_hanging_itag=True)
                span_labels_pred[text_id].extend([(task_label, start, end, self._hf_model_label) for start, end in spans])
        return [span_labels_pred[get_doc_id(doc)] for doc in X]

if __name__ == '__main__':
    pass
