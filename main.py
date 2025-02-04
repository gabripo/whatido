from src.fine_tuning_hf.TrainingDatasetHF import TrainingDatasetHF
from src.fine_tuning_hf.SFTHF import SupervisedFineTuningHF

if __name__ == "__main__":
    dataset = TrainingDatasetHF("SecchiAlessandro/dataset-email-screenshots", split="train")

    instruction = "You are an expert corporate manager. Make a sensitivity analysis of what you see in this image and give some advices on how to improve."
    dataset.convert_to_conversation(conversion_instruction=instruction)
    # print(dataset.dataset[0]) # debug print
    # print(dataset.converted_dataset[0]) # debug print

    # sft = SupervisedFineTuningHF('xtuner/llava-llama-3-8b-v1_1-transformers')
    sft = SupervisedFineTuningHF("llava-hf/llava-1.5-7b-hf")
    sft.build_model()
    sft.set_processor()
    sft.load_dataset(dataset)
    # print(sft.processor.apply_chat_template(sft.train_data[0]['messages'], tokenize=False))
    sft.set_trainer()
    sft.train()