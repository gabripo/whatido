import os
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch

class HFModel:
    def __init__(
            self,
            model_name: str = None,
            base_model_name: str = None,
            model_load_config: dict = None,
            quant_config: BitsAndBytesConfig = None
            ):
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.model_load_config = model_load_config
        self.quant_config = quant_config
        self.model = None
        self.base_model = None
        self.tokenizer = None
        self.model_local_path = None

    def get_hf_model(self, local_save: bool = False):
        if self.model_name is None:
            print(f"Invalid base or derived model name!")
            return

        
        self.set_model_load_config()

        if self.is_model_peft():
            self.set_quantization_config()

            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **self.model_load_config,
                )
            self.model = PeftModel.from_pretrained(self.base_model, self.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **self.model_load_config,
                )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if local_save:
            self.model_local_path = os.path.abspath(os.getcwd(), self.model_name)
            if self.is_model_peft():
                self.base_model.save_pretrained(self.model_local_path)
            self.model.save_pretrained(self.model_local_path)
            self.tokenizer.save_pretrained(self.model_local_path)

    def set_quantization_config(
            self,
            load_in_4_bit: bool = True,
            bnb_4bit_quant_type: str ="nf4",
            bnb_4bit_compute_dtype: type =torch.float16,
            bnb_4bit_use_double_quant: bool =True,
            **kwargs
            ):
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4_bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            **kwargs
        )
        if self.quant_config is not None:
            self.model_load_config['quantization_config'] = self.quant_config

    def set_model_load_config(
            self,
            device_map: str = "auto",
            trust_remote_code: bool = True,
            token: bool = False,
            **kwargs
            ):
        self.model_load_config = {
            'device_map': device_map,
            'trust_remote_code': trust_remote_code,
            'token': token,
            **kwargs
        }

    def is_model_peft(self):
        return self.base_model_name is not None