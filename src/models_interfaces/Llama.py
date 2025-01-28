import ollama

class LlamaQuery:
    def __init__(
            self,
            model_name: str,
            messages: list,
            images: list[str] | None = None):
        self.model_name = model_name
        self.messages = messages
        self.images = images

    def query(
            self,
            query_text: str,
            include_history: bool = True,
            add_to_history: bool = True,
            llamakwargs: dict = {}):
        new_query = {
            'role': 'user',
            'content': query_text,
        }
        if self.images is not None:
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

# Factory Pattern
class LlamaQueryFactory:
    @staticmethod
    def create_text_query(
        history: list = [],
        model_name: str = "llama3.2"
        ) -> LlamaQuery:
        return LlamaQuery(
            model_name=model_name,
            messages=history,
        )
    
    @staticmethod
    def create_vision_query(
        images: list[str] = [],
        history: list = [],
        model_name: str = "llama3.2-vision"
        ) -> LlamaQuery:
        return LlamaQuery(
            model_name=model_name,
            messages=history,
            images=images
        )