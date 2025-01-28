import ollama

class LlamaQuery:
    def __init__(self, **kwargs: dict):
        required_obj_vars = ["model_name", "messages"]
        for var in required_obj_vars:
            setattr(self, var, kwargs.get(var, None))
            kwargs.pop(var)
        
        self.__dict__.update(kwargs)

    def query(self, query_text: str, include_history: bool = True, add_to_history: bool = True, llamakwargs: dict = {}):
        new_query = {
            'role': 'user',
            'content': query_text,
        }
        if hasattr(self, 'images'):
            new_query.update({'images': self.images})
        
        if include_history:
            self.messages.append(new_query)
        else:
            self.messages = [new_query]

        response = ollama.chat(
            model = self.model_name,
            messages = self.messages,
            **llamakwargs
        )

        if add_to_history:
            self.messages.append(response.message)
        return response.message

class LlamaVisionQuery(LlamaQuery):
    def __init__(self, images: list[str], history: list[dict] = [], model_name: str = "llama3.2-vision"):
        super().__init__(
            messages=history,
            model_name=model_name,
            images=images,
        )

    def query(self, query_text: str, include_history: bool = True, add_to_history: bool = True):
        return super().query(
            query_text=query_text,
            include_history=include_history,
            add_to_history=add_to_history
        )

class LlamaTextQuery(LlamaQuery):
    def __init__(self, history: list[dict] = [], model_name: str = "llama3.2"):
        super().__init__(
            messages=history,
            model_name=model_name,
            )

    def query(self, query_text: str, include_history: bool = True, add_to_history: bool = True, llamakwargs: dict = {}):
        return super().query(
            query_text=query_text,
            include_history=include_history,
            add_to_history=add_to_history,
            llamakwargs=llamakwargs
        )