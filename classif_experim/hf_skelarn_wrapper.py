from logging import Logger
import random
import sys
import tempfile
from abc import ABCMeta, abstractmethod
from copy import copy
from functools import partial
from pathlib import Path
import os, pickle
from typing import List

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    TextClassificationPipeline, EarlyStoppingCallback, TrainerCallback
from transformers import AutoTokenizer, set_seed
from transformers import DataCollatorWithPadding

from classif_experim.pynvml_helpers import print_gpu_utilization, print_cuda_devices

class LoggingCallback(TrainerCallback):
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info(f'Epoch {state.epoch} ended')

def set_torch_np_random_rseed(rseed: int) -> None:
    """
    Set random seeds for numpy, random, and torch to ensure reproducibility.
    
    Args:
        rseed (int): The random seed value.

    Returns:
        None
    """
    np.random.seed(rseed)
    random.seed(rseed)
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)

class SklearnTransformerBase(metaclass=ABCMeta):
    def __init__(self, hf_model_label: str, lang: str, eval: float = 0.1, learning_rate: float = 2e-5, num_train_epochs: int = 3, 
                 weight_decay: float = 0.01, batch_size: int = 16, warmup: float = 0.1, gradient_accumulation_steps: int = 1, 
                 max_seq_length: int = 128, device: torch.device = None, rnd_seed: int = 381757, tmp_folder: str = None,
                 fc_up_layers: int = 0, fc_down_layers: int = 1, eval_metric: str = None, rel_stop_threshold: float = 0.01,
                 stop_patience = 3, logger: Logger = None) -> None:
        """
        Initialize the SklearnTransformerBase with the given parameters.
        
        Args:
            hf_model_label (str): Hugging Face model identifier.
            lang (str): Language of the model ('en' for English, 'es' for Spanish).
            eval (float, optional): Proportion of the training set used for evaluation. Default is 0.1.
            learning_rate (float, optional): Learning rate for training. Default is 2e-5.
            num_train_epochs (int, optional): Number of training epochs. Default is 3. Won't be used if stop_metric is not None.
            weight_decay (float, optional): Weight decay for training. Default is 0.01.
            batch_size (int, optional): Batch size for training. Default is 16.
            warmup (float, optional): Warmup ratio for learning rate. Default is 0.1.
            gradient_accumulation_steps (int, optional): Gradient accumulation steps. Default is 1.
            max_seq_length (int, optional): Maximum sequence length for inputs. Default is 128.
            device (torch.device, optional): Device for model training and evaluation. Default is None.
            rnd_seed (int, optional): Random seed for reproducibility. Default is 381757.
            tmp_folder (str, optional): Temporary folder for model checkpoints. Default is None.
            fc_up_layers (int, optional): Number of up layers in the custom top module. Default is 0.
            fc_down_layers (int, optional): Number of down layers in the custom top module. Default is 1.
            eval_metric (str, optional): Metric to use for early stopping. Default is None so no early stopping is performed.
            rel_stop_threshold (float, optional): Relative threshold for early stopping. Default is 0.01.
            stop_patience (int, optional): Patience for early stopping. Default is 3.
            logger (Logger, optional): Logger for the class. Default is None.

        Returns:
            None
        """

        self._hf_model_label = hf_model_label
        self._learning_rate = learning_rate; self._num_train_epochs = num_train_epochs
        self._eval_metric = eval_metric; self._rel_stop_threshold = rel_stop_threshold; self._stop_patience = stop_patience
        self._weight_decay = weight_decay
        self._eval = eval; self._lang = lang
        if device: self._device = device
        else: self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._max_seq_length = max_seq_length
        self._tmp_folder = tmp_folder
        self._rnd_seed = rnd_seed
        self._batch_size = batch_size
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._warmup = warmup
        self.tokenizer = None
        self.model = None
        self._logger = logger

        self.fc_up_layers = fc_up_layers
        self.fc_down_layers = fc_down_layers
        
        #set_seed(rnd_seed)
        set_torch_np_random_rseed(rnd_seed)

    def _init_temp_folder(self) -> None:
        """
        Initialize the temporary folder for model training checkpoints.

        Args:
            None

        Returns:
            None
        """

        if self._tmp_folder is None:
            self._tmp_folder_object = tempfile.TemporaryDirectory()
            self._tmp_folder = self._tmp_folder_object.name
        else:
            assert Path(self._tmp_folder).exists() # todo do assert alway, create exception
        print(f'Temporary folder: {self._tmp_folder}')

    def _cleanup_temp_folder(self) -> None:
        """
        Clean up the temporary folder used for model training checkpoints.

        Args:
            None

        Returns:
            None
        """

        if hasattr(self, '_tmp_folder_object'):
            self._tmp_folder_object.cleanup()
            del self._tmp_folder_object
            self._tmp_folder = None # new fit will initiate new tmp. folder creation
        else: # leave the user-specified tmp folder intact
            pass

    def _init_train_args(self) -> None:
        """
        Initialize Hugging Face TrainingArguments for model training.

        Args:
            None

        Returns:
            None
        """

        if self._eval is None:
            save_params = {
                'save_strategy' : 'no',
                'eval_strategy' : 'no',
                'output_dir': self._tmp_folder,
                'num_train_epochs': self._num_train_epochs,
            }
        else:
            save_params = {
                'output_dir' : self._tmp_folder,
                'save_strategy' : 'epoch',
                'eval_strategy' : 'epoch',
                'save_total_limit' : 2,
                'load_best_model_at_end' : True,
                'metric_for_best_model' : self._eval_metric if self._eval_metric is not None else None,
                'greater_is_better' : True if self._eval_metric in ['f1_macro', 'f1', 'f1_neg' 'accuracy', 'precision', 'recall', 'mcc'] else None,
                'num_train_epochs' : self._num_train_epochs if self._eval_metric is None else sys.maxsize,
                'lr_scheduler_type' : 'reduce_lr_on_plateau',
                'lr_scheduler_kwargs': {
                    'mode': 'max',
                    'factor': 0.1,
                    'patience': self._stop_patience,
                    'threshold': self._rel_stop_threshold,
                },
            }
        self._training_args = TrainingArguments(
            do_train=True, do_eval=self._eval is not None,
            learning_rate=self._learning_rate, 
            warmup_ratio=self._warmup, weight_decay=self._weight_decay,
            per_device_train_batch_size=self._batch_size,
            per_device_eval_batch_size=self._batch_size,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            overwrite_output_dir=True, resume_from_checkpoint=False,
            **save_params
        )

    @abstractmethod
    def fit(self, X: List[str], y: List[str]) -> None:
        """
        Train the model with the provided texts and labels.
        
        Args:
            X (List[str]): List of texts for training.
            y (List[str]): List of labels corresponding to the texts.

        Returns:
            None
        """

        pass

    @abstractmethod
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict the labels of the provided texts.
        
        Args:
            X (List[str]): List of texts to be classified.

        Returns:
            np.ndarray: Array of predicted labels.
        """

        pass

    def __del__(self):
        if hasattr(self, 'model') and self.model is not None: del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            del self.tokenizer_params
        self._cleanup_temp_folder()
        torch.cuda.empty_cache()

    @property
    def device(self): return self._device

    @device.setter
    def device(self, dev):
        self._device = dev

    def save(self, output_dir: str) -> None:
        """
        Save the model, tokenizer, and class configuration to the output directory.
        
        Args:
            output_dir (str): Directory to save the model and tokenizer.

        Returns:
            None
        """

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        # save model and tokenizer
        #model_path = os.path.join(output_dir, 'pytorch_model.bin')
        #torch.save(self.model.state_dict(), model_path)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.save_class_attributes(output_dir)

    ATTRIBUTES_FILE_NAME = 'class_attributes.pkl'

    def save_class_attributes(self, output_dir: str) -> None:
        """
        Save the class attributes to the output directory, excluding 'model' and 'tokenizer'.
        
        Args:
            output_dir (str): Directory to save the class attributes.

        Returns:
            None
        """

        attributes_path = os.path.join(output_dir, self.ATTRIBUTES_FILE_NAME)
        # Save class attributes except 'model' and 'tokenizer'
        # TODO add non-serializable attributes to the list, enable sub-class customization
        with open(attributes_path, 'wb') as attributes_file:
            attributes_to_save = self.__dict__.copy()
            attributes_to_save.pop('model', None)
            attributes_to_save.pop('tokenizer', None)
            pickle.dump(attributes_to_save, attributes_file)

    @classmethod
    def load_class_attributes(cls, input_dir: str) -> 'SklearnTransformerBase':
        """
        Load class attributes from the specified directory, excluding 'model' and 'tokenizer'.
        
        Args:
            input_dir (str): Directory to load the class attributes from.

        Returns:
            SklearnTransformerBase: Instance of the class with loaded attributes.
        """

        attributes_path = os.path.join(input_dir, cls.ATTRIBUTES_FILE_NAME)
        with open(attributes_path, 'rb') as attributes_file:
            attributes = pickle.load(attributes_file)
        instance = cls.__new__(cls)
        instance.__dict__.update(attributes)
        return instance
    

class CustomTop(torch.nn.Module):
    def __init__(self, input_size, num_classes, fc_up_layers, fc_down_layers):
        super(CustomTop, self).__init__()
        self.fc_up_layers = fc_up_layers
        self.fc_down_layers = fc_down_layers
        self.fc_up = torch.nn.ModuleList()
        self.fc_down = torch.nn.ModuleList()
        
        for i in range(self.fc_up_layers):
            self.fc_up.append(torch.nn.Linear(input_size, input_size * 2))
            self.fc_up.append(torch.nn.ReLU())
            self.fc_up.append(torch.nn.Dropout(0.1))
            input_size *= 2
        
        for i in range(self.fc_down_layers):
            self.fc_down.append(torch.nn.Linear(input_size, input_size // 2))
            self.fc_down.append(torch.nn.ReLU())
            self.fc_down.append(torch.nn.Dropout(0.1))
            input_size //= 2
        
        self.fc_out = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x[:, 0, :]
        for fc in self.fc_up:
            x = fc(x)
        
        for fc in self.fc_down:
            x = fc(x)
        
        x = self.fc_out(x)
        return x

class SklearnTransformerClassif(SklearnTransformerBase):
    """
    Adapter of Hugging Face transformers to scikit-learn interface.
    The workflow is load model, fine-tune, apply and/or save.
    """

    def _init_tokenizer_params(self) -> None:
        """
        Initialize tokenizer parameters for truncation and maximum sequence length.

        Args:
            None

        Returns:
            None
        """

        if not hasattr(self, 'tokenizer_params'):
            self.tokenizer_params = {'truncation': True}
            if self._max_seq_length is not None: self.tokenizer_params['max_length'] = self._max_seq_length

    def _init_model(self, num_classes: int) -> None:
        """
        Load model and tokenizer for classification fine-tuning.
        
        Args:
            num_classes (int): Number of classes for the classification task.

        Returns:
            None
        """

        # load and init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_label)
        self._init_tokenizer_params()
        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
                        self._hf_model_label, num_labels=num_classes)
        
        # print('Existing model top:')
        # print(self.model.classifier)
        is_roberta = True
        if is_roberta:
            custom_top = CustomTop(self.model.config.hidden_size, num_classes, self.fc_up_layers, self.fc_down_layers)
        else:
            custom_top = self._create_custom_top_module(self.model.config.hidden_size, num_classes)
        
        if custom_top is not None:
            self.model.classifier = custom_top
        
        # print('New model top:')
        # print(self.model.classifier)
        
        # print('Model top loaded with {} up layers and {} down layers'.format(self.fc_up_layers, self.fc_down_layers))
        # print('Model top: {}'.format(custom_top))
        self.model = self.model.to(self._device)
        

    def _create_custom_top_module(self, input_size, num_classes: int) -> torch.nn.Module:
        """
        Create a custom top layer for the model with the given number of classes.
        
        Args:
            num_classes (int): Number of classes for the classification task.

        Returns:
            torch.nn.Module: Custom top layer for the model.
        """
        assert self.fc_up_layers >= 0 and self.fc_down_layers >= 0
        if self.fc_up_layers == 0 and self.fc_down_layers == 0:
            return None

        top = torch.nn.Sequential()

        for i in range(self.fc_up_layers):
            # print('fc_up{}: {} -> {}'.format(i, input_size * (2 ** i), input_size * (2 ** (i+1))) )
            top.add_module(
                f'fc_up{i}',
                torch.nn.Linear(
                    input_size * (2 ** i),
                    input_size * (2 ** (i+1))
                )
            )
            top.add_module(f'act_up{i}', torch.nn.ReLU())
            top.add_module(f'drop_up{i}', torch.nn.Dropout(0.1))
    
        for i in range(self.fc_down_layers):
            # print('fc_down{}: {} -> {}'.format(i, input_size * (2 ** (self.fc_up_layers - i)), input_size * (2 ** (self.fc_up_layers - i - 1))) )
            top.add_module(
                f'fc_down{i}',
                torch.nn.Linear(

                    int(input_size * (2 ** (self.fc_up_layers - i))),

                    int(input_size * (2 ** (self.fc_up_layers - i - 1)))
                )
            )
            top.add_module(f'act_down{i}', torch.nn.ReLU())
            top.add_module(f'drop_down{i}', torch.nn.Dropout(0.1))
        
        top.add_module('fc_out',
            torch.nn.Linear(
                int(input_size * (2 ** (self.fc_up_layers - self.fc_down_layers))),
                num_classes
                )
            )
        return top
    

    def set_string_labels(self, labels: List[str]) -> None:
        """
        Set a 1-to-1 mapping between string labels and corresponding integer indices.
        For binary classification, the labels therefore should be ['NEGATIVE LABEL', 'POSITIVE LABEL'].

        Args:
            labels (List[str]): List of string labels for the classification task.

        Returns:
            None
        """

        assert len(labels) == len(set(labels)) # assert that the labels are unique
        self._str_labels = labels

    def _init_classes(self, labels: List[str]) -> None:
        """
        Initialize class label data from the labels of the training set.
        
        Args:
            labels (List[str]): List of labels from the training set.

        Returns:
            None
        """

        if not hasattr(self, '_str_labels'): # induce labels from the input list of (train) labels
            self._class_labels = sorted(list(set(l for l in labels)))
        else:
            self._class_labels = copy(self._str_labels)
        self._num_classes = len(self._class_labels)
        self._cls_ix2label = { ix: l for ix, l in enumerate(self._class_labels) }
        self._cls_label2ix = { l: ix for ix, l in enumerate(self._class_labels) }

    def _labels2indices(self, labels: List[str]) -> np.ndarray:
        """
        Map class labels in input format to numbers in [0, ... , NUM_CLASSES]

        Args:
            labels (List[str]): List of class labels.

        Returns:
            np.ndarray: Array of label indices.
        """

        return np.array([ix for ix in map(lambda l: self._cls_label2ix[l], labels)])

    def _indices2labels(self, indices: np.ndarray) -> np.ndarray:
        """
        Map class indices in [0,...,NUM_CLASSES] to original class labels

        Args:
            indices (np.ndarray): Array of label indices.

        Returns:
            np.ndarray: Array of class labels.
        """

        return np.array([l for l in map(lambda ix: self._cls_ix2label[ix], indices)])

    def _prepare_dataset(self, X: List[str], y: List[str]) -> DatasetDict:
        """
        Convert fit() params to hugginface-compatible datasets.Dataset

        Args:
            X (List[str]): List of texts for training.
            y (List[str]): List of labels corresponding to the texts.

        Returns:
            DatasetDict: Hugging Face Dataset containing training and evaluation splits.
        """

        int_labels = self._labels2indices(y)
        df = pd.DataFrame({'text': X, 'label': int_labels})
        if self._eval:
            train, eval = \
                train_test_split(df, test_size=self._eval, random_state=self._rnd_seed, stratify=df[['label']])
            dset = DatasetDict(
                {'train': datasets.Dataset.from_pandas(train), 'eval': datasets.Dataset.from_pandas(eval)})
        else:
            dset = datasets.Dataset.from_pandas(df)
        return dset

    def fit(self, X: List[str], y: List[str]) -> None:
        """
        Train the model with the provided texts and labels.
        
        Args:
            X (List[str]): List of texts for training.
            y (List[str]): List of labels corresponding to the texts.

        Returns:
            None
        """

        # delete old model from tmp folder, if it exists
        self._init_classes(y)
        # model and tokenizer init
        self._init_model(self._num_classes)
        self._init_temp_folder()
        self._do_training(X, y)
        self._cleanup_temp_folder()
        # input txt formatting and tokenization
        # training

    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict the labels of the provided texts.
        
        Args:
            X (List[str]): List of texts to be classified.

        Returns:
            np.ndarray: Array of predicted labels.
        """

        #todo X 2 pandas df, df to Dataset.from_pandas dset ? or simply from iterable ?
        dset = datasets.Dataset.from_list([{'text': txt} for txt in X])
        pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=self._device,
                                          max_length=self._max_seq_length, truncation=True, batch_size=32)
        result = pipe(dset['text'], function_to_apply='softmax')
        del pipe
        torch.cuda.empty_cache()
        # parse predictions, map to original labels
        #todo regex-based extraction of integers from the specific format
        pred = [int(r['label'][-1]) for r in result] # assumes *LABEL$N format
        return self._indices2labels(pred)

    def tokenize(self, txt: str, **kwargs) -> dict:
        """
        Tokenize the input text using the model's tokenizer.
        
        Args:
            txt (str): Input text to be tokenized.
            **kwargs: Additional parameters for tokenization.

        Returns:
            dict: Tokenized input.
        """
        self._init_tokenizer_params()
        # joint self.tokenizer_params and kwargs
        params = self.tokenizer_params.copy()
        for k, v in kwargs.items(): params[k] = v
        return self.tokenizer(txt, **params)

    def _compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation.

        Args:
            eval_pred (tuple): A tuple (predictions, labels) where predictions are logits and labels are indices of the true labels.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # Calculate metrics
        f1_macro = f1_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='binary', pos_label=1)  # Assuming 1 is the positive class
        f1_neg = f1_score(labels, predictions, average='binary', pos_label=0)  # Assuming 0 is the negative class
        acc = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='binary', pos_label=1)
        recall = recall_score(labels, predictions, average='binary', pos_label=1)
        mcc = matthews_corrcoef(labels, predictions)

        metrics = {
            "f1_macro": f1_macro,
            "f1": f1,
            "f1_neg": f1_neg,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "mcc": mcc
        }
    
        if self._logger is not None:
            self._logger.info(f"Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])}")

        return metrics

    def _do_training(self, X: List[str], y: List[str]) -> None:
        """
        Perform the training process for the model.
        
        Args:
            X (List[str]): List of texts for training.
            y (List[str]): List of labels corresponding to the texts.

        Returns:
            None
        """
        torch.manual_seed(self._rnd_seed)
        def preprocess_function(examples):
            return self.tokenizer(examples['text'], **self.tokenizer_params)
        dset = self._prepare_dataset(X, y)
        tokenized_dset = dset.map(preprocess_function, batched=True)
        self._init_train_args()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        if self._eval: train, eval = tokenized_dset['train'], tokenized_dset['eval']
        else: train, eval = tokenized_dset, None
        trainer = Trainer(
            model=self.model,
            args=self._training_args,
            train_dataset=train,
            eval_dataset=eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self._stop_patience,
                    early_stopping_threshold=self._rel_stop_threshold
                ) if self._eval_metric is not None else None,
                LoggingCallback(self._logger) if self._logger is not None else None,
            ] if self._eval else []
        )
        trainer.train()
        if self.model is not trainer.model: # just in case
            del self.model
            self.model = trainer.model
        del trainer
        torch.cuda.empty_cache()
        

    @classmethod
    def load(cls, input_dir: str, device: torch.device = None) -> 'SklearnTransformerClassif':
        """
        Load the model, tokenizer, and class configuration from the input directory.
        
        Args:
            input_dir (str): Directory to load the model and tokenizer from.
            device (torch.device, optional): Device to load the model onto. Default is None.

        Returns:
            SklearnTransformerClassif: Loaded model instance.
        """
        instance = cls.load_class_attributes(input_dir)
        # load tokenizer and model
        # TODO move tokenizer loading to superclass?
        #tokenizer_path = os.path.join(input_dir, 'tokenizer_config.json')
        instance.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        model_path = os.path.join(input_dir, 'pytorch_model.bin')
        if device is None: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        instance.model = AutoModelForSequenceClassification.from_pretrained(input_dir).to(device)

        return instance


def test_hf_wrapper(test_dset: str, model: str = 'bert-base-uncased', device: str = 'cuda:0', subsample: int = 500, 
                    rnd_seed: int = 4140, eval: float = 0.1) -> None:
    """
    Test Hugging Face transformer wrapper with a given dataset.
    
    Args:
        test_dset (str): Dataset identifier from Hugging Face datasets library.
        model (str, optional): Hugging Face model identifier. Default is 'bert-base-uncased'.
        device (str, optional): Device for model training and evaluation. Default is 'cuda:0'.
        subsample (int, optional): Number of samples to use for testing. Default is 500.
        rnd_seed (int, optional): Random seed for reproducibility. Default is 4140.
        eval (float, optional): Proportion of the training set used for evaluation. Default is 0.1.

    Returns:
        None
    """    # prepare test dataset
    dset = load_dataset(test_dset)
    texts = np.array(dset['train']['text'])
    labels = np.array(dset['train']['label'])
    if subsample:
        random.seed(rnd_seed)
        ixs = random.sample(range(len(texts)), subsample)
        texts, labels = texts[ixs], labels[ixs]
    txt_trdev, txt_test, lab_trdev, lab_test = \
        train_test_split(texts, labels, test_size=0.8, random_state=rnd_seed, stratify=labels)
    # train model, evaluate
    tr = SklearnTransformerClassif(num_train_epochs=5, eval=eval, hf_model_label=model, rnd_seed=rnd_seed, device=device,
                                   lang='en')
    
    tr.fit(txt_trdev, lab_trdev)
    lab_pred = tr.predict(txt_test)
    f1 = partial(f1_score, average='binary')
    acc = accuracy_score
    print(f'f1: {f1(lab_test, lab_pred):.3f}, acc: {acc(lab_test, lab_pred):.3f}')

if __name__ == '__main__':
    test_hf_wrapper(test_dset='imdb', subsample=100, eval=None)
