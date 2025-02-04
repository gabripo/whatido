import torch
from abc import ABC, abstractmethod
from .TrainingDatasetPytorch import TrainingDatasetPytorch

class FineTraining(ABC):
    @abstractmethod
    def load_dataset(self, dataset: TrainingDatasetPytorch):
        pass

    def set_device(self):
        if torch.backends.mps.is_available():
            # Apple Metal support
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load_tuned_model(self):
        pass

    @abstractmethod
    def print_training_characteristics(self):
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