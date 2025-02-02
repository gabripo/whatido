from src.fine_tuning.TrainingDatasetHF import TrainingDatasetHF

if __name__ == "__main__":
    dataset = TrainingDatasetHF("SecchiAlessandro/dataset-email-screenshots", split="train")

    instruction = "You are an expert corporate manager. Make a sensitivity analysis of what you see in this image and give some advices on how to improve."
    dataset.convert_to_conversation(conversion_instruction=instruction)
    print(dataset.converted_dataset[0]) # debug print