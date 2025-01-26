import torch
from abc import abstractmethod
from .TrainingDataset import TrainingDataset

class FineTraining:
    def __init__(self):
        self.device = None
        self.loss = {
            'train': 0,
            'test': 0,
        }

    @abstractmethod
    def load_dataset(self, dataset: TrainingDataset):
        pass

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

# Command Pattern
class TrainingCommander:
    def __init__(self):
        pass

    def execute(self, query):
        pass

# Chain of Responsibility Pattern
class TrainingHandler:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def handle(self, query):
        for handler in self.handlers:
            handler.execute(query)