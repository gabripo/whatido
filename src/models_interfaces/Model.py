from abc import abstractmethod

class ModelQuery:
    @abstractmethod
    def query(self, *args, **kwargs):
        pass

class ModelQueryFactory:
    @staticmethod
    @abstractmethod
    def create_text_query(*args, **kwargs) -> ModelQuery:
        pass

    @staticmethod
    @abstractmethod
    def create_vision_query(*args, **kwargs) -> ModelQuery:
        pass