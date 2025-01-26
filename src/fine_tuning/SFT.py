from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os
from torch.utils.data import DataLoader
from .FineTraining import FineTraining
from .TrainingDataset import TrainingDataset, TOKENIZER_MAX_LENGTH
from ..database_manager.Score import MAX_SCORE, MIN_SCORE, Score

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_TOKENIZER_NAME = "auto-tokenizer"
DEFAULT_OPTIMIZER_NAME = "adamw"
DEFAULT_NUM_EPOCHS = 1
MAX_SCORE_NORMALIZED = 1
MIN_SCORE_NORMALIZED = 0

class SupervisedFineTraining(FineTraining):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, tokenizer_name: str = DEFAULT_TOKENIZER_NAME, optimizer_name: str = DEFAULT_OPTIMIZER_NAME, split_size: float = 0.2, random_state: int = 1):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.optimizer_name = optimizer_name
        self.tuned_model_name = model_name + "_fine_tuned"

        self.device = None
        self.model = None
        self.tuned_model = None
        self.tokenizer = None
        self.tuned_tokenizer = None
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
        self.loss_function = torch.nn.MSELoss()
        self.loss = {
            'train': [],
            'test': [],
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
        self._reset_loss()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1} : Training started")
            self.loss['train'].append(self._train(self.loaders["train"]))
            self.loss['test'].append(self._validate(self.loaders["test"]))
            print(f"Epoch {epoch + 1} : Train loss {self.loss['train'][-1]:.4f} | Test loss {self.loss['test'][-1]:.4f}")
        print(f"Training for {self.__class__.__name__} concluded!")
        self.has_trained = True

    def _reset_loss(self):
        self.loss['train'] = []
        self.loss['test'] = []

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
            labels = self._normalize(batch["labels"]).to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            loss = self.loss_function(logits, labels)
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
                labels = self._normalize(batch["labels"]).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                loss = self.loss_function(logits, labels)
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
    
    def _normalize(self, tensor: torch.Tensor, max_val: float = float(MAX_SCORE_NORMALIZED), min_val: float = float(MIN_SCORE_NORMALIZED)) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        
        curr_max = tensor.max(dim=1, keepdim=True).values
        curr_min = tensor.min(dim=1, keepdim=True).values
        mask = (curr_min == curr_max)
        curr_min[mask] = MIN_SCORE_NORMALIZED
        curr_max[mask] = MAX_SCORE_NORMALIZED
        
        tensor_normalized = min_val + (tensor - curr_min) * (max_val - min_val) / (curr_max - curr_min)
        return tensor_normalized
    
    def _denormalize(self, array_normalized: list[float], max_val: float = float(MAX_SCORE), min_val: float = float(MIN_SCORE)) -> torch.Tensor:
        if len(array_normalized) == 0:
            return array_normalized
        
        curr_max = max(array_normalized)
        curr_min = min(array_normalized)
        if curr_min == curr_max:
            return array_normalized
        
        array = min_val + (array_normalized - curr_min) * (max_val - min_val) / (curr_max - curr_min)
        return array
        
    def infer(self, input: str = None) -> list:
        if input is None or type(input) != str:
            print("Provide a valid input in the form of a string.")
            return
        
        self.load_tuned_model(self.tuned_model_name)
        if not self.has_trained:
            print("Fine-tuned model not available, train it before trying to infere.")
            return

        inputs = self.tuned_tokenizer(
            input,
            return_tensors="pt",
            max_length=TOKENIZER_MAX_LENGTH,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.tuned_model.eval()
        with torch.no_grad():
            outputs = self.tuned_model(**inputs)
            scores = self._denormalize(outputs.logits.cpu().numpy().flatten())
        categories_scores = dict(zip(Score._default_score_names(), scores))
        return categories_scores
    
    def save(self, tuned_model_name: str = None):
        if tuned_model_name is None:
            tuned_model_name = self.tuned_model_name
        else:
            self.tuned_model_name = tuned_model_name

        if self.has_trained:
            self.model.save_pretrained(f'./{self.tuned_model_name}')
            self.tokenizer.save_pretrained(f'./{self.tuned_model_name}')

    def load_tuned_model(self, tuned_model_name: str = None):
        if tuned_model_name is None:
            tuned_model_name = self.tuned_model_name
        
        tuned_model_path = os.path.abspath(tuned_model_name)
        if os.path.exists(tuned_model_path):
            try:
                self.tuned_model = AutoModelForSequenceClassification.from_pretrained(tuned_model_name).to(self.device)
                self.tuned_tokenizer = AutoTokenizer.from_pretrained(tuned_model_name)
                self.has_trained = True
            except:
                print(f"Impossible to load tuned model with name {tuned_model_name}")

    def print_training_characteristics(self):
        print(f"Model: {self.model_name}")
        print(f"Tokenizer: {self.tokenizer_name}")
        print(f"Optimizer: {self.optimizer_name}")
        if self.has_trained:
            print(f"Loss (train): {self.loss['train']}")
            print(f"Loss (test): {self.loss['test']}")
            print(f"Tuned model: {self.tuned_model_name}")
        