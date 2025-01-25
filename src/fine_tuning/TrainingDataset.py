import os, json, torch
from torch.utils.data import Dataset

TOKENIZER_MAX_LENGTH = 128

class TrainingDataset(Dataset):
    def __init__(self, database_path: str, tokenizer, max_length: int = TOKENIZER_MAX_LENGTH):
        self.database_path = database_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = None
        self.import_data()
        
    def import_data(self):
        try:
            self._import_data_from_json(self.database_path)
        except:
            print(f"Unsupported format for dataset {self.database_path} ! No data will be loaded.\n")

    def _import_data_from_json(self, json_file_path: str):
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                try:
                    self.data = json.load(json_file)
                except:
                    print(f"Error while loading json file {json_file_path} !\n")
        else:
            print(f"File {json_file_path} not available!\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            return self._load_email_score(item)
        except:
            print(f"Impossiblo to load item at index {idx}.\n")
    
    def _load_email_score(self, item: dict):
        loaded_item = self._empty_item()

        if "email" in item.keys():
            text = item["email"]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            loaded_item["input_ids"] = encoding["input_ids"].flatten()
            loaded_item["attention_mask"] = encoding["attention_mask"].flatten()

        if "score" in item.keys():
            loaded_item["labels"] = torch.tensor(list(item["score"].values()), dtype=torch.float)

        return loaded_item
    
    def _empty_item(self):
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }