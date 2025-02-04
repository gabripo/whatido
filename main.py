from src.fine_tuning_hf.TrainingDatasetHF import TrainingDatasetHF
from src.models_interfaces.ModelHF import ModelHF

try:
    from src.models_interfaces.Unsloth import UnslothVisionModel
    UNSLOTH_SUPPORTED = True
except:
    UNSLOTH_SUPPORTED = False

if __name__ == "__main__":
    dataset = TrainingDatasetHF("SecchiAlessandro/dataset-email-screenshots", split="train")

    instruction = "You are an expert corporate manager. Make a sensitivity analysis of what you see in this image and give some advices on how to improve."
    dataset.convert_to_conversation(conversion_instruction=instruction)
    # print(dataset.dataset[0]) # debug print
    # print(dataset.converted_dataset[0]) # debug print

    if UNSLOTH_SUPPORTED:
        model = UnslothVisionModel(
            model_name="gabripo/lora_model_productivity_vision",
            )
        model.enable_inference()
    else:
        model = ModelHF(
            model_name="gabripo/lora_model_productivity_vision",
            base_model_name="unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit",
            )
        model.get_hf_model(local_save=True)
    # TODO implement inference pipeline