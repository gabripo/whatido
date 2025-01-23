import os, asyncio, json
from abc import abstractmethod

# Singleton Pattern
class Database:
    _instance = None

    def __new__(cls, name: str = "db", path: str="database", *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.database_name = cls.__name__
            cls._instance.database_path = path
            cls._instance.create_folder()
            cls._instance.generated_data = {}
            cls._instance.files = []
        return cls._instance
       
    @abstractmethod
    def create_folder(self) -> None:
        self.database_path = os.path.join(os.getcwd(), self.database_path)
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"Folder {self.database_path} for database {self.database_name}\n")
        else:
            print(f"Folder {self.database_path} already exists!\n")

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def store(self, store_single_queries: bool = False):
        io_tasks = [self._dump_json_data(self.database_name, self.generated_data)]
        if store_single_queries:
            for query_responses in self.generated_data:
                query = query_responses['query']
                filename = query[:10] + f"_{hash(query)}"
                io_tasks.append(self._dump_json_data(filename, query_responses))
        
        self._execute_event_loop_gather(io_tasks)
    
    def _execute_event_loop_gather(self, tasks: list):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop=event_loop)
        try:
            event_loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            event_loop.close()

    async def _dump_json_data(self, filename: str, data, indent_size: int = 4):
        filename_full_path = os.path.join(self.database_path, filename + '.json')
        with open(filename_full_path, 'w') as json_file:
            json.dump(data, json_file, indent=indent_size)


    @abstractmethod
    def print(self):
        print(f"Files in the database {self.__class__.__name__}:\n")
        for filename in self.files:
            print(f"{filename}")