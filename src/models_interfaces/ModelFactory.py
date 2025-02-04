from .ModelQuery import ModelQueryFactory
from .Llama import LlamaQueryFactory

class ModelFactory:
    @staticmethod
    def create_model_factory(model_family: str = 'llama') -> ModelQueryFactory:
        if model_family == 'llama':
            return LlamaQueryFactory()
        else:
            print(f"Unsupported model family {model_family}.")