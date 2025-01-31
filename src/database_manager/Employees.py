from .Database import Database

class EmployeesDatabase(Database):
    def __init__(self, name: str, path: str = "database"):
        self.database_name = name
        self.database_path = path

    def create_folder(self):
        return super().create_folder()
    
    def get_database_abspath(self):
        return super().get_database_abspath()
    
    def build(self):
        super().build()
        self._compute_profitability()

    def _compute_profitability(self):
        pass
    
    def store(self):
        return super().store()
    
    def store_single_entry(self, entry):
        return super().store_single_entry(entry)
    
    def clear(self):
        return super().clear()
    
    def print(self):
        return super().print()