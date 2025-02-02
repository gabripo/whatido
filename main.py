import os
from src.database_manager.Emails import DatabaseEmails
# from src.database_manager.Employees import EmployeesDatabase
# from src.database_manager.ActivityScreeenshots import ScreenshotsDatabase
from src.fine_tuning.TrainingDataset import TrainingDataset
from src.fine_tuning.SFT import SupervisedFineTraining
# from src.fine_tuning.LORA import LORA

TRAIN_EMAILS_MODEL = False

if __name__ == "__main__":
    db_emails = DatabaseEmails('gen_emails')
    sft = SupervisedFineTraining()
    if TRAIN_EMAILS_MODEL:
        dataset_path = db_emails.get_database_abspath()
        dataset = TrainingDataset(dataset_path, sft.tokenizer)
        sft.load_dataset(dataset)
        sft.train(num_epochs=10)
        sft.save()
        sft.print_training_characteristics()
    infered_score = sft.infer("Subject: Sorry\nObject: I am sorry. I cannot speak German.")
    print(infered_score)

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