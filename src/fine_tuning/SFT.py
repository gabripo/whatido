from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader
from .FineTraining import FineTraining
from .TrainingDataset import TrainingDataset

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_TOKENIZER = "auto-tokenizer"

class SupervisedFineTraining(FineTraining):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, tokenizer = DEFAULT_TOKENIZER, split_size: float = 0.2, random_state: int = 1):
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
        self.model_name = None
        self.set_model_name(model_name)
        self.tokenizer = None
        self.set_tokenizer(tokenizer)
        self.num_labels = 0
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_options = {
            'lr': 5e-5
        }
        self.optimizer = None

    def load_dataset(self, dataset: TrainingDataset):
        self.train_data = dict(zip(self.train_data.keys(), train_test_split(dataset, **self.split_options)))
        self.loaders["train"] = DataLoader(self.train_data['X_train'], **self.loaders_options)
        self.loaders["test"] = DataLoader(self.train_data['X_test'], **self.loaders_options)
        self.num_labels = dataset.num_labels
    
    def train(self):
        return super().train()
    
    def validate(self):
        return super().validate()
    
    def build_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)
        
    def set_model_name(self, model_name: str = DEFAULT_MODEL_NAME):
        if not model_name is None:
            self.model_name = model_name
            print(f"Model {model_name} chosen for {self.__class__.__name__}\n")
        else:
            print(f"No model selected for {self.__class__.__name__}\n")

    def set_tokenizer(self, tokenizer: str = DEFAULT_TOKENIZER):
        if self.model_name is None:
            print(f"Invalid model name {self.model_name} for {self.__class__.__name__} ! No tokenizer can be selected.\n")
            return
        
        supported_tokenizers = {
            'auto-tokenizer': AutoTokenizer.from_pretrained(self.model_name)
        }
        if not tokenizer in supported_tokenizers:
            print(f"Specified tokenizer {tokenizer} unsupported! No tokenizer for {self.__class__.__name__}\n")
            return
        self.tokenizer = supported_tokenizers[tokenizer]
    
    def set_optimizer(self, optimizer_name: str = 'adamw'):    
        supported_optimizers = {
            'adamw': AdamW(self.model.parameters(), **self.optimizer_options),
        }
        if not optimizer_name in supported_optimizers:
            print(f"Specified optimizer {optimizer_name} unsupported! No optimizer for {self.__class__.__name__} will be set.\n")
            return
        self.optimizer = supported_optimizers[optimizer_name]