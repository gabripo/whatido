import os, asyncio, json, shutil
from abc import abstractmethod

# Singleton Pattern
class Database:
    _instance = None

    def __new__(cls, name: str = "db", path: str = None, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance.database_name = cls.__name__
            cls._instance.database_path = path
            cls._instance.create_folder()
            cls._instance.store_single_queries = False
            cls._instance.json_dump_settings = {
                'indent': 4,
                'default': lambda obj: obj.__json__()
            }
            cls._instance.generated_data = {}
            cls._instance.store_while_generating = True
            cls._instance.file_lock = asyncio.Lock()
            cls._instance.files = set()
        return cls._instance
       
    @abstractmethod
    def create_folder(self) -> None:
        if self.database_path is None:
            self.database_path = "database"
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
    def store(self):
        io_tasks = [self._dump_json_data(self.database_name, self.generated_data)]
        if self.store_single_queries:
            for query_responses in self.generated_data:
                query = query_responses['query']
                filename = query[:10] + f"_{hash(query)}"
                io_tasks.append(self._dump_json_data(filename, query_responses))
        
        self._execute_event_loop_gather(io_tasks)

    @abstractmethod
    def store_single_entry(self, entry: dict):
        filename_full_path = os.path.join(self.database_path, self.database_name + '.json')
        self._dump_json_data_single(filename_full_path, [entry])

    @abstractmethod
    def clear(self):
        for file_path in self.files:
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    def _execute_event_loop_gather(self, tasks: list):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop=event_loop)
        try:
            event_loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            event_loop.close()

    async def _dump_json_data(self, filename: str, data: list):
        filename_full_path = os.path.join(self.database_path, filename + '.json')
        for data_single in data:
            self._dump_json_data_single(filename_full_path, data_single)

    def _dump_json_data_single(self, filename_full_path: str, data: list):
        file_access_type = 'r+' if self._is_json_readable(filename_full_path) else 'w'
        with open(filename_full_path, file_access_type) as json_file:
            if file_access_type != 'w' and not self._is_json_file_empty(filename_full_path):
                existing_data = json.load(json_file)
                data.extend(existing_data)  # new data on the top
                json_file.seek(0)
            json.dump(data, json_file, **self.json_dump_settings)

        if os.path.exists(filename_full_path) and filename_full_path not in self.files:
            self.files.add(filename_full_path)

    def _is_json_readable(self, filename_full_path: str) -> bool:
        if os.path.exists(filename_full_path):
            try:
                with open(filename_full_path, 'r') as json_file:
                    json.load(json_file)
                return True
            except:
                print(f"File {filename_full_path} is not a valid JSON file!\n")
        return False

    def _is_json_file_empty(self, file_full_path: str) -> bool:
        if not os.path.exists(file_full_path):
            return True
        with open(file_full_path, 'r') as file:
            file.seek(0, 2)
            return file.tell() == 0


    @abstractmethod
    def print(self):
        print(f"Files in the database {self.database_name}:\n")
        for filename in self.files:
            print(f"{filename}")