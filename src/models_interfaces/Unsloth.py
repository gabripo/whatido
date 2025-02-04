from unsloth import FastVisionModel
from .ModelHF import ModelHF

SUPPORTED_VISION_MODELS_4BIT = {
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
 } # More models at https://huggingface.co/unsloth

class UnslothVisionModel(ModelHF):
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pretrained_settings = {
            'load_in_4bit': True,
            'use_gradient_checkpointing': "unsloth",
        }
        self.peft_settings = {
            'finetune_vision_layers': False, # False if not finetuning vision layers
            'finetune_language_layers': True, # False if not finetuning language layers
            'finetune_attention_modules': True, # False if not finetuning attention layers
            'finetune_mlp_modules': True, # False if not finetuning MLP layers
            'r': 16,           # The larger, the higher the accuracy, but might overfit
            'lora_alpha': 16,  # Recommended alpha == r at least
            'lora_dropout': 0,
            'bias': "none",
            'random_state': 3407,
            'use_rslora': False,  # We support rank stabilized LoRA
            'loftq_config': None, # And LoftQ
            # 'target_modules': "all-linear", # Optional now! Can specify a list if needed
        }

    def build_from_pretrained(self):
        if not self._check_model_name_available:
            return
        if self.model_name not in SUPPORTED_VISION_MODELS_4BIT:
            print(f"Model {self.model_name} not supported.")
            return
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.model_name,
            **self.pretrained_settings
        )
    
    def peft_model(self):
        if not self._check_model_available:
            return
        
        self.model = FastVisionModel.get_peft_model(
            self.model,
            **self.peft_settings
        )

    def enable_training(self):
        if not self._check_model_available:
            return
        FastVisionModel.for_training(self.model)

    def enable_inference(self):
        if not self._check_model_available:
            return
        FastVisionModel.for_inference(self.model)

    def _check_model_name_available(self):
        if self.model_name is None:
            print(f"Model name not available for {self.__class__.__name__}!")
            return False
        return True
    
    def _check_model_available(self):
        if self.model is None:
            print(f"Model not available for {self.__class__.__name__}!")
            return False
        return True
