from .FineTraining import FineTraining

DEFAULT_MODEL_NAME = "llama3.2-vision"

class LORA(FineTraining):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name

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