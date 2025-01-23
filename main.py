import os
from src.chat_handler.Query import QueryHistory, QueryCommander, QueryHandler
from src.screenshots_manager.Screenshot import ScreenshotManager
from src.database_manager.Emails import DatabaseEmails

model_name = 'llama3.2-vision'
ANALYZE_SCREEN = True
ANALYZE_ASSETS = True
TEST_TEXT_QUERY = True

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

    de = DatabaseEmails('gen_emails', 'test')
    queries = [
        'write an aggressive e-mail, max 100 characters',
        'write an e-mail explaining low-pass filtering, max 300 characters',
        'write an e-mail explaining the usage of \'epoustoflant\' in French with an example, max 300 characters',
        'scrivi una e-mail scusandoti di un evento grave accaduto in azienda, max 200 caratteri e in Italiano'
        ]
    de.build(queries, 2)
    de.store()