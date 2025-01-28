import os
from src.chat_handler.Query import QueryHistory, QueryCommander, QueryHandler
from src.screenshots_manager.Screenshot import ScreenshotManager
from src.database_manager.Emails import DatabaseEmails, QueryScorePair
from src.fine_tuning.TrainingDataset import TrainingDataset
from src.fine_tuning.SFT import SupervisedFineTraining
from src.Economics.Productivity_scores import get_efficiency_scores

# Call the function to get the efficiency scores
p_s = get_efficiency_scores()

ANALYZE_SCREEN = False
ANALYZE_ASSETS = False
TEST_TEXT_QUERY = False
GENERATE_EMAILS = False

if __name__ == "__main__":
    history = QueryHistory()
    handler = QueryHandler()

    image_query = QueryCommander(query_history=history)
    handler.add_handler(image_query)
    
    if ANALYZE_SCREEN:
        screenshot_manager = ScreenshotManager()
        screenshot_manager.set_scale_factor(1)
        screenshot_manager.capture_screenshot()
        handler.handle({'images': screenshot_manager.screenshots, 'query': 'What do you see on my screen?'})
        history.print_history()
        history.clear_history()

    if ANALYZE_ASSETS:
        queries = [
            {'images': [os.path.join(os.getcwd(), 'assets', 'horsehead_nebula.jpg')], 'query': 'What is in this image?'},
            {'images': [os.path.join(os.getcwd(), 'assets', 'Crab_MultiChandra_960.jpg')], 'query': 'What is in this image?'},
            {'images': [], 'query': 'Are the objects in the images in the same galaxy cluster?'}
        ]
        for query in queries:
            handler.handle(query)
        history.print_history()
        history.clear_history()

    if TEST_TEXT_QUERY:
        handler.handle({'query': 'what are you?'})
        history.print_history()
        history.clear_history()

    if GENERATE_EMAILS:
        de = DatabaseEmails('gen_emails')
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
    dataset_path = 'database/gen_emails.json'
    dataset = TrainingDataset(dataset_path, sft.tokenizer)
    sft.load_dataset(dataset)
    sft.train(num_epochs=10)
    sft.save()
    sft.print_training_characteristics()

    infered_score = sft.infer("Subject: Sorry\nObject: I am sorry. I cannot speak German.")
    print(infered_score)