import os
# from src.database_manager.Employees import EmployeesDatabase
# from src.database_manager.ActivityScreeenshots import ScreenshotsDatabase
from src.fine_tuning.TrainingDataset import TrainingDataset
# from src.fine_tuning.LORA import LORA

if __name__ == "__main__":
    pass
    # ds = ScreenshotsDatabase('screenshots')
    # # TODO generate / entry database
    # ds.build()
    # ds.print()

    # lora = LORA("llava-llama3")
    # dataset = TrainingDataset(ds.get_database_abspath())
    # lora.load_dataset(dataset)
    # # TODO fine-tuning of model
    # lora.train()
    # lora.save()
    # lora.print_training_characteristics()
    # # TODO create a valid input for inference
    # img_path = os.path.abspath(os.path.join(os.getcwd(), 'screenshots', 'test.png'))
    # infered_score = lora.infer(img_path)
    # print(infered_score)

    # demp = EmployeesDatabase('employee_data')
    # demp.print()