import datasets
import io
from PIL import Image

class TrainingDatasetHF(datasets.Dataset):
    def __init__(self, hg_repo_id: str, split: str = "train"):
        try:
            self.dataset = datasets.load_dataset(hg_repo_id, split=split)
        except Exception as exc:
            print(f"Dataset load {hg_repo_id} failed: {exc}")
            self.dataset = None
        self.converted_dataset = None

    def convert_to_conversation(self, conversion_instruction: str = ""):
        if not self.dataset:
            return None
        
        converted_dataset = [
            self._convert_to_conversation_sample(
                sample=sample,
                conversion_instruction=conversion_instruction
                ) 
            for sample in self.dataset
        ]
        self.converted_dataset = converted_dataset

    @staticmethod
    def _convert_to_conversation_sample(sample, conversion_instruction: str = ""):
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : conversion_instruction},
                {"type" : "image", "image" : sample["image"]} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : sample["metrics"]} ]
            },
        ]
        return { "messages" : conversation }