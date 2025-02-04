from .FineTuningHF import FineTuningHF

class SupervisedFineTuningHF(FineTuningHF):
    def __init__(self):
        super().__init__()

    def load_dataset(self, dataset):
        return super().load_dataset(dataset)

    def train(self):
        return super().train()
    
    def infer(self):
        return super().infer()
    
    def save(self):
        return super().save()
    
    def load_tuned_model(self):
        return super().load_tuned_model()
    
    def print_training_characteristics(self):
        return super().print_training_characteristics()