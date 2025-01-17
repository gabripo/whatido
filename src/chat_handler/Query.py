from ..models_interfaces.Llama import LlamaQuery

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

# Command Pattern
class QueryCommander:
    def __init__(self, query_history: QueryHistory):
        self.query_history = query_history

    def execute(self, query):
        history = self.query_history.get_history()
        query_obj = LlamaQuery(query['images'], history)

        response = query_obj.query(query['query'])

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