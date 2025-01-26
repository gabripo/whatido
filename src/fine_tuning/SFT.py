from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from .FineTraining import FineTraining
from .TrainingDataset import TrainingDataset

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_TOKENIZER_NAME = "auto-tokenizer"
DEFAULT_OPTIMIZER_NAME = "adamw"
DEFAULT_NUM_EPOCHS = 1

class SupervisedFineTraining(FineTraining):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, tokenizer_name: str = DEFAULT_TOKENIZER_NAME, optimizer_name: str = DEFAULT_OPTIMIZER_NAME, split_size: float = 0.2, random_state: int = 1):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.optimizer_name = optimizer_name
        self.tuned_model_name = model_name + "_fine_tuned"

        self.device = None
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        self.split_options = {
            'test_size': split_size,
            'random_state': random_state
        }
        self.train_data = {
            'X_train': [],
            'X_test': [],
            'y_train': [],
            'y_test': [],
        }
        self.loaders_options = {
            'batch_size': 8,
            'shuffle': True,
        }
        self.loaders = {
            'train': None,
            'test': None,
        }
        self.num_labels = 0
        self.optimizer_options = {
            'lr': 5e-5
        }
        self.loss = {
            'train': 0,
            'test': 0,
        }
        self.has_trained = False

        self.set_device()
        self.set_tokenizer()

    def load_dataset(self, dataset: TrainingDataset):
        self.train_data = dict(zip(self.train_data.keys(), train_test_split(dataset, **self.split_options)))
        self.loaders["train"] = DataLoader(self.train_data['X_train'], **self.loaders_options)
        self.loaders["test"] = DataLoader(self.train_data['X_test'], **self.loaders_options)
        self.num_labels = dataset.num_labels

    def build_model(self):
        if self.model_name is None:
            print(f"No model selected for {self.__class__.__name__}\n")
            return
        if self.device is None:
            print(f"Device not properly set for {self.__class__.__name__} ! Impossible to send model {self.model_name}.\n")
            return
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)

    def set_tokenizer(self, tokenizer_name: str = None):
        if self.model_name is None:
            print(f"Invalid model name {self.model_name} for {self.__class__.__name__} ! No tokenizer can be selected.\n")
            return
        if self.tokenizer_name is None:
            print(f"No tokenizer specified for {self.__class__.__name__}.\n")
            return
        if not tokenizer_name is None:
            self.tokenizer_name = tokenizer_name
        
        supported_tokenizers = {
            'auto-tokenizer': AutoTokenizer.from_pretrained(self.model_name)
        }
        if not self.tokenizer_name in supported_tokenizers:
            print(f"Specified tokenizer {self.tokenizer_name} unsupported! No tokenizer for {self.__class__.__name__}\n")
            return
        self.tokenizer = supported_tokenizers[self.tokenizer_name]
    
    def set_optimizer(self, optimizer_name: str = None):
        if self.model is None or not hasattr(self.model, 'parameters'):
            print(f"No model loaded in {self.__class__.__name__} : the optimizer cannot be set.\n")
            return
        if self.optimizer_name is None:
            print(f"No optimizer specified for {self.__class__.__name__}.\n")
            return
        if not optimizer_name is None:
            self.optimizer_name = optimizer_name

        supported_optimizers = {
            'adamw': torch.optim.Adam(self.model.parameters(), **self.optimizer_options),
        }
        if not self.optimizer_name in supported_optimizers:
            print(f"Specified optimizer {self.optimizer_name} unsupported! No optimizer for {self.__class__.__name__} will be set.\n")
            return
        self.optimizer = supported_optimizers[self.optimizer_name]
    
    def train(self, num_epochs: int = DEFAULT_NUM_EPOCHS):
        print(f"Start training for {self.__class__.__name__} ...")
        self.build_model()
        self.set_optimizer()
        for epoch in range(num_epochs):
            self.loss['train'] = self._train(self.loaders["train"])
            self.loss['test'] = self._validate(self.loaders["test"])
            print(f"Epoch {epoch + 1} : Train loss {self.loss['train']:.4f} | Test loss {self.loss['test']:.4f}")
        print(f"Training for {self.__class__.__name__} concluded!")
        self.has_trained = True

    def _train(self, dataloader: DataLoader = None) -> float:
        if dataloader is None:
            print(f"Invalid dataloader specified for the training. No model training in {self.__class__.__init__} will be performed.\n")
            return 0
        if not self._can_train():
            print(f"Impossible to train with {self.__class__.__name__}.\n")
            return 0
        
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            self.optimizer.zero_grad()

            if not self._is_batch_valid(batch):
                print("Invalid batch, skipping the loop this time...")
                continue
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _validate(self, dataloader: DataLoader = None) -> float:
        if dataloader is None:
            print(f"Invalid dataloader specified for the validation. No model validation in {self.__class__.__init__} will be performed.\n")
            return 0

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                if not self._is_batch_valid(batch):
                    print("Invalid batch, skipping the loop this time...")
                    continue
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item()
        return total_loss / len(dataloader)

    def _can_train(self) -> bool:
        checks = {
            'device': self.device,
            'tokenizer': self.tokenizer,
            'model': self.model,
            'optimizer': self.optimizer,
        }
        for name, value in checks.items():
            if value is None:
                print(f"No {name} available for {self.__class__.__name__}")
        
        return all(value is not None for value in checks.values())
    
    def _is_batch_valid(self, batch: dict) -> bool:
        supported_names = {
            'input_ids': False,
            'attention_mask': False,
            'labels': False,
        }
        for name, value in batch.items():
            if value is None or len(value) == 0:
                print(f"Empty value for {name} : nothing will be transferred to device.")
            if name in supported_names:
                supported_names[name] = True
        
        return all(value is True for value in supported_names.values())
        
    def save(self, tuned_model_name: str = None):
        if tuned_model_name is None:
            tuned_model_name = self.tuned_model_name
        else:
            self.tuned_model_name = tuned_model_name

        if self.has_trained:
            self.model.save_pretrained(f'./{self.tuned_model_name}')
            self.tokenizer.save_pretrained(f'./{self.tuned_model_name}')