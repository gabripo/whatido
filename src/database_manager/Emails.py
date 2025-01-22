import os
from .Database import Database

class DatabaseEmails(Database):
    def __init__(self, path: str):
        self.database_path = path

    def create_folder(self) -> None:
        return super().create_folder()

    def build(self):
        return super().build()
    
    def store(self):
        return super().store()
    
    def print(self):
        return super().print()