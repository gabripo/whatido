import copy
from ..models_interfaces.ModelFactory import ModelFactory

# Singleton Pattern
class QueryHistory:
    _instance = None

    def __new__(cls):
        # prevent creation of other objects other than the already existing one
        if cls._instance is None:
            cls._instance = super(QueryHistory, cls).__new__(cls)
            cls._instance.history = []
        return cls._instance
    
    def add_message(self, message):
        self.history.append(message)
    
    def get_history(self):
        return self.history
    
    def clear_history(self):
        self.history = []

    def print_history(self):
        for query in self.history:
            print(f"role : {query["role"]}")
            print(f"content : {query["content"]}")

# Command Pattern
class QueryCommander:
    def __init__(self, query_history: QueryHistory):
        self.query_history = query_history

    def execute(self, query: dict, model_family: str = 'llama'):
        factory_obj = ModelFactory.create_model_factory(model_family=model_family)
        history = copy.deepcopy(self.query_history.get_history())
        if 'images' in query:
            query_obj = factory_obj.create_vision_query(images=query['images'], history=history, model_name='llava-llama3')
        else:
            query_obj = factory_obj.create_text_query(history=history)
        
        response = query_obj.query(query_text=query['query'])

        self.query_history.add_message(response)

# Chain of Responsibility Pattern
class QueryHandler:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def handle(self, query):
        for handler in self.handlers:
            handler.execute(query)