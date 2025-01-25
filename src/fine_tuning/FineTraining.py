from abc import abstractmethod
from .TrainingDataset import TrainingDataset

class FineTraining:
    @abstractmethod
    def load_dataset(self, dataset: TrainingDataset):
        pass
    
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