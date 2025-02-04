import datasets
from PIL import Image

class TrainingDatasetHF(datasets.Dataset):
    def __init__(self, hg_repo_id: str, split: str = "train"):
        try:
            self.dataset = datasets.load_dataset(hg_repo_id, split=split)
        except Exception as exc:
            print(f"Dataset load {hg_repo_id} failed: {exc}")
            self.dataset = None
        self.converted_dataset = None
        self.name = hg_repo_id

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
            {
                "role": "user",
                "content" : [
                    {"type" : "text",  "text"  : conversion_instruction},
                    {"type" : "image"}
                ]
            },
            {
                "role" : "assistant",
                "content" : [
                    {"type" : "text",  "text"  : str(sample['metrics'])} # cast sample['metrics'] as a string TODO make it a structured output
                ]
            },
        ]
        return {
            "messages" : conversation,
            "images": sample["image"],
            }
    
if __name__ == "__main__":
    dataset = TrainingDatasetHF("SecchiAlessandro/dataset-email-screenshots", split="train")
    print(dataset.dataset[0])

    instruction = "You are an expert corporate manager. Make a sensitivity analysis of what you see in this image and give some advices on how to improve."
    dataset.convert_to_conversation(conversion_instruction=instruction)
    print(dataset.converted_dataset[0])

if __name__ == "__main__":
    import _setup_test_env