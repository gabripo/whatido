from src.fine_tuning.TrainingDatasetHF import TrainingDatasetHF
from src.models_interfaces.HFModel import HFModel

if __name__ == "__main__":
    dataset = TrainingDatasetHF("SecchiAlessandro/dataset-email-screenshots", split="train")

    instruction = "You are an expert corporate manager. Make a sensitivity analysis of what you see in this image and give some advices on how to improve."
    dataset.convert_to_conversation(conversion_instruction=instruction)
    # print(dataset.dataset[0]) # debug print
    # print(dataset.converted_dataset[0]) # debug print

    model = HFModel(
        "unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit",
        "gabripo/lora_model_productivity_vision"
        )
    model.get_hf_model(local_save=True)