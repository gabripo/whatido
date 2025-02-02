import os
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch

class HFModel:
    def __init__(self, model_name: str = None, base_model_name: str = None):
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.model = None
        self.base_model = None
        self.tokenizer = None
        self.model_local_path = None

    def get_hf_model(self, local_save: bool = False):
        if self.model_name is None:
            print(f"Invalid base or derived model name!")
            return
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        if self.base_model_name is not None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quant_config,
                device_map="auto",  # Automatically distribute layers across devices
                trust_remote_code=True,  # Required for custom models
                # token=True,  # Needed for private models
                )
        self.model = PeftModel.from_pretrained(self.base_model, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if local_save:
            self.model_local_path = os.path.abspath(os.getcwd(), self.model_name)
            if self.base_model_name is not None:
                self.base_model.save_pretrained(self.model_local_path)
            self.model.save_pretrained(self.model_local_path)
            self.tokenizer.save_pretrained(self.model_local_path)