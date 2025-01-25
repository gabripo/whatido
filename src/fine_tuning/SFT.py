from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from .FineTraining import FineTraining
from .TrainingDataset import TrainingDataset

class SupervisedFineTraining(FineTraining):
    def __init__(self, split_size: float = 0.2, random_state: int = 1):
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
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_dataset(self, dataset: TrainingDataset):
        self.train_data = dict(zip(self.train_data.keys(), train_test_split(dataset, **self.split_options)))
    
    def train(self):
        return super().train()
    
    def validate(self):
        return super().validate()