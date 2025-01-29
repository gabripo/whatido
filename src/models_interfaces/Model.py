from abc import ABC, abstractmethod

class ModelQuery(ABC):
    @abstractmethod
    def query(self, *args, **kwargs):
        pass

class ModelQueryFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_text_query(*args, **kwargs) -> ModelQuery:
        pass

    @staticmethod
    @abstractmethod
    def create_vision_query(*args, **kwargs) -> ModelQuery:
        pass