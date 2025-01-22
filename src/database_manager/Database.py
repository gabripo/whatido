import os
from abc import abstractmethod

# Singleton Pattern
class Database:
    _instance = None

    def __new__(cls, path: str="database", *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.database_path = path
            cls._instance.create_folder()
            cls._instance.files = []
        return cls._instance
       
    @abstractmethod
    def create_folder(self) -> None:
        self.database_path = os.path.join(os.getcwd(), self.database_path)
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"Folder {self.database_path} for database {self.__class__.__name__}\n")
        else:
            print(f"Folder {self.database_path} already exists!\n")

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def print(self):
        pass