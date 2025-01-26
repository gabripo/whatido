from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader
from .FineTraining import FineTraining
from .TrainingDataset import TrainingDataset

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_TOKENIZER_NAME = "auto-tokenizer"
DEFAULT_OPTIMIZER_NAME = "adamw"

class SupervisedFineTraining(FineTraining):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, tokenizer_name: str = DEFAULT_TOKENIZER_NAME, optimizer_name: str = DEFAULT_OPTIMIZER_NAME, split_size: float = 0.2, random_state: int = 1):
        self.split_options = {
            'test_size': split_size,
            'random_state': random_state
        }
        self.train_data = {
            'X_train': [],
            'X_test': [],
            'y_train': [],
            'y_test': [],
        }
        self.loaders_options = {
            'batch_size': 8,
            'shuffle': True,
        }
        self.loaders = {
            'train': None,
            'test': None,
        }
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.optimizer_name = optimizer_name

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        self.num_labels = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_options = {
            'lr': 5e-5
        }

    def load_dataset(self, dataset: TrainingDataset):
        self.train_data = dict(zip(self.train_data.keys(), train_test_split(dataset, **self.split_options)))
        self.loaders["train"] = DataLoader(self.train_data['X_train'], **self.loaders_options)
        self.loaders["test"] = DataLoader(self.train_data['X_test'], **self.loaders_options)
        self.num_labels = dataset.num_labels

    def build_model(self):
        if self.model_name is None:
            print(f"No model selected for {self.__class__.__name__}\n")
            return
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)

    def set_tokenizer(self, tokenizer_name: str = None):
        if self.model_name is None:
            print(f"Invalid model name {self.model_name} for {self.__class__.__name__} ! No tokenizer can be selected.\n")
            return
        if self.tokenizer_name is None:
            print(f"No tokenizer specified for {self.__class__.__name__}.\n")
            return
        if not tokenizer_name is None:
            self.tokenizer_name = tokenizer_name
        
        supported_tokenizers = {
            'auto-tokenizer': AutoTokenizer.from_pretrained(self.model_name)
        }
        if not self.tokenizer_name in supported_tokenizers:
            print(f"Specified tokenizer {self.tokenizer_name} unsupported! No tokenizer for {self.__class__.__name__}\n")
            return
        self.tokenizer = supported_tokenizers[self.tokenizer_name]
    
    def set_optimizer(self, optimizer_name: str = None):
        if self.model is None or not hasattr(self.model, 'parameters'):
            print(f"No model loaded in {self.__class__.__name__} : the optimizer cannot be set.\n")
            return
        if self.optimizer_name is None:
            print(f"No optimizer specified for {self.__class__.__name__}.\n")
            return
        if not optimizer_name is None:
            self.optimizer_name = optimizer_name

        supported_optimizers = {
            'adamw': AdamW(self.model.parameters(), **self.optimizer_options),
        }
        if not self.optimizer_name in supported_optimizers:
            print(f"Specified optimizer {self.optimizer_name} unsupported! No optimizer for {self.__class__.__name__} will be set.\n")
            return
        self.optimizer = supported_optimizers[self.optimizer_name]
    
    def train(self):
        self.set_tokenizer()
        self.build_model()
        self.set_optimizer()
    
    def validate(self):
        return super().validate()