import ollama

class LlamaVisionQuery:
    def __init__(self, images: list[str], history: list[dict], model_name: str = "llama3.2-vision"):
        self.images = images
        self.model_name = model_name
        self.messages = history

    def query(self, query_text: str, include_history: bool = True):
        new_query = {
            'role': 'user',
            'content': query_text,
            'images' : self.images
        }
        if include_history:
            self.messages.append(new_query)
        else:
            self.messages = new_query
        
        response = ollama.chat(
            model = self.model_name,
            messages = self.messages
        )

        return response.message

class LlamaTextQuery:
    def __init__(self, history: list[dict], model_name: str = "llama3.2"):
        self.model_name = model_name
        self.messages = history

    def query(self, query_text: str, include_history: bool = True):
        new_query = {
            'role': 'user',
            'content': query_text,
        }
        if include_history:
            self.messages.append(new_query)
        else:
            self.messages = new_query
        
        response = ollama.chat(
            model = self.model_name,
            messages = self.messages
        )

        return response.message