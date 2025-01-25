from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from .FineTraining import FineTraining
from .TrainingDataset import TrainingDataset

class SupervisedFineTraining(FineTraining):
    def __init__(self, model_name: str = None, tokenizer = None, split_size: float = 0.2, random_state: int = 1):
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
        self.model_name = "bert-base-uncased" if model_name is None else None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) if tokenizer is None else None
        self.num_labels = 0
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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