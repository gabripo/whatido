import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from trl import TrlParser, ScriptArguments, SFTConfig, ModelConfig, get_quantization_config, get_kbit_device_map, SFTTrainer
from .FineTuningHF import FineTuningHF
from .TrainingDatasetHF import TrainingDatasetHF

class SupervisedFineTuningHF(FineTuningHF):
    def __init__(
            self,
            model_name: str = None,
            output_dir: str = None
            ):
        self.model_name = model_name
        self.tuned_model_name = self.model_name + "_fine_tuned" if self.model_name is not None else None
        self.output_dir = output_dir if output_dir is not None else "./fine_tuned" # TODO smarter way to determine it

        self.model = None
        self.tuned_model = None
        self.processor = None
        self.trainer = None

        self.train_data = None
        self.train_data_name = None
        self.script_args = None
        self.training_args = None
        self.model_args = None
        self.model_kwargs = {}

        self.set_device()
        self.determine_args()

    def load_dataset(
            self,
            dataset: TrainingDatasetHF,
            load_converted: bool = True,
            ):
        try:
            if load_converted:
                self.train_data = dataset.converted_dataset
            else:
                self.train_data = dataset.dataset
            self.train_data_name = dataset.name
        except Exception as exc:
            print(f"Impossible to load the dataset! Exception was {exc}")
            return
        
        if self.train_data is None:
            print(f"Empty dataset!")

    def build_model(self):
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                trust_remote_code=self.model_args.trust_remote_code,
                **self.model_kwargs
            )
        except Exception as exc:
            print(f"Impossible to load the model {self.model_name} . Exception was {exc}")

    def set_processor(self):
        if self._is_model_name_valid():
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.model_args.trust_remote_code,
                device_map='auto',
            )
        else:
            print(f"Impossible to set processor due to invalid model name!")

    def set_trainer(self):
        if self.training_args is None:
            print(f"Impossible to set trainer: no training arguments!")
            return
        
        try:
            self.trainer = SFTTrainer(
                model=self.model,
                args=self.training_args,
                data_collator=self.collate_fn,
                train_dataset=self.train_data, # TODO introduce the eval_dataset by split
                processing_class=self.processor.tokenizer,
            )
        except Exception as exc:
            print(f"Impossible to set the trainer for the model {self.model_name} ! Exception was {exc}")

    def collate_fn(self, examples):
        # Get the texts and images, and apply the chat template
        texts = [self.processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]
        if isinstance(self.model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    def train(self):
        if self.trainer is None:
            print(f"No available trainer!")
            return
        self.trainer.train()
    
    def infer(self):
        return super().infer()
    
    def save(self):
        if self.trainer is not None:
            self.trainer.save_model(self.training_args.output_dir)
    
    def load_tuned_model(self):
        return super().load_tuned_model()
    
    def print_training_characteristics(self):
        return super().print_training_characteristics()
    
    def determine_args(self):
        # parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
        # self.script_args, self.training_args, self.model_args = parser.parse_args_and_config()

        # self.training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
        # self.training_args.remove_unused_columns = False
        # self.training_args.dataset_kwargs = {"skip_prepare_dataset": True}

        # torch_dtype = (
        #     self.model_args.torch_dtype
        #     if self.model_args.torch_dtype in ["auto", None]
        #     else getattr(torch, self.model_args.torch_dtype)
        # )
        # quantization_config = get_quantization_config(self.model_args)
        # self.model_kwargs = dict(
        #     revision=self.model_args.model_revision,
        #     attn_implementation=self.model_args.attn_implementation,
        #     torch_dtype=torch_dtype,
        #     device_map=get_kbit_device_map() if quantization_config is not None else None,
        #     quantization_config=quantization_config,
        # )
        self.script_args = ScriptArguments(dataset_name=self.train_data_name)

        self.training_args = SFTConfig(output_dir=self.output_dir)
        self.training_args.remove_unused_columns = False
        self.training_args.dataset_kwargs = {"skip_prepare_dataset": True}

        self.model_args = ModelConfig(model_name_or_path=self.model_name)
        self.model_args.trust_remote_code = True

    def _is_model_name_valid(self) -> bool:
        return self.model_name is not None