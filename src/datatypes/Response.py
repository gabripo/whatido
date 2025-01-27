from pydantic import BaseModel
from abc import abstractmethod

class Response(BaseModel):
    @abstractmethod
    def get_content(self):
        pass
    
    @abstractmethod
    def print(self):
        pass
    
class EmailResponse(Response):
    email_subject: str
    email_text: str | list[str] | list[list[str]]

    def get_content(self):
        return {
            'email_subject': self.email_subject,
            'email_text': self.email_text,
        }
    
    def __json__(self):
        return self.get_content()
    
    def print(self):
        for key, value in self.content.items():
            print(f"key: {key} | value: {value}")