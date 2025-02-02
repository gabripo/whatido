import os
from src.database_manager.Emails import DatabaseEmails, QueryScorePair
from src.database_manager.Employees import EmployeesDatabase
from src.database_manager.ActivityScreeenshots import ScreenshotsDatabase
from src.fine_tuning.TrainingDataset import TrainingDataset
from src.fine_tuning.SFT import SupervisedFineTraining
from src.fine_tuning.LORA import LORA
# from src.Economics.Productivity_scores import get_efficiency_scores	
	# Call the function to get the efficiency scores
# p_s = get_efficiency_scores()

GENERATE_EMAILS = False
TRAIN_EMAILS_MODEL = False

if __name__ == "__main__":
    de = DatabaseEmails('gen_emails')
    if GENERATE_EMAILS:
        queries = [
            QueryScorePair('write an aggressive e-mail', [10, 10, 0, 10, 0]),
            QueryScorePair('write an e-mail explaining low-pass filtering', [10, 3, 10, 0, 2]),
            QueryScorePair('write an e-mail explaining the usage of epoustoflant in French with an example', [6, 10, 0, 0, 7]),
            QueryScorePair('scrivi una e-mail scusandoti di un evento grave accaduto in azienda in Italiano', [0, 9, 0, 0, 10]),
            QueryScorePair('schreib eine aggressive E-Mail auf Deutsch',[0, 10, 0, 10, 0]),
            ]
        de.build(queries, 20)
        de.print()

    sft = SupervisedFineTraining()
    if TRAIN_EMAILS_MODEL:
        dataset_path = de.get_database_abspath()
        dataset = TrainingDataset(dataset_path, sft.tokenizer)
        sft.load_dataset(dataset)
        sft.train(num_epochs=10)
        sft.save()
        sft.print_training_characteristics()
    infered_score = sft.infer("Subject: Sorry\nObject: I am sorry. I cannot speak German.")
    print(infered_score)

    ds = ScreenshotsDatabase('screenshots')
    # TODO generate / entry database
    ds.build()
    ds.print()

    lora = LORA("llava-llama3")
    dataset = TrainingDataset(ds.get_database_abspath())
    lora.load_dataset(dataset)
    # TODO fine-tuning of model
    lora.train()
    lora.save()
    lora.print_training_characteristics()
    # TODO create a valid input for inference
    img_path = os.path.abspath(os.path.join(os.getcwd(), 'screenshots', 'test.png'))
    infered_score = lora.infer(img_path)
    print(infered_score)

    demp = EmployeesDatabase('employee_data')
    demp.print()