import os
from src.chat_handler.Query import QueryHistory, QueryCommander, QueryHandler
from src.screenshots_manager.Screenshot import ScreenshotManager

model_name = 'llama3.2-vision'
ANALYZE_SCREEN = True
ANALYZE_ASSETS = True

if __name__ == "__main__":
    history = QueryHistory()
    handler = QueryHandler()

    image_query = QueryCommander(query_history=history)
    handler.add_handler(image_query)
    
    if ANALYZE_SCREEN:
        screenshot_manager = ScreenshotManager()
        screenshot_manager.set_scale_factor(0.3)
        screenshot_manager.capture_screenshot()
        handler.handle({'images': screenshot_manager.screenshots, 'query': 'What do you see on my screen?'})
        print(history.get_history())
        history.clear_history()

    if ANALYZE_ASSETS:
        queries = [
            {'images': [os.path.join(os.getcwd(), 'assets', 'horsehead_nebula.jpg')], 'query': 'What is in this image?'},
            {'images': [os.path.join(os.getcwd(), 'assets', 'Crab_MultiChandra_960.jpg')], 'query': 'What is in this image?'},
            {'images': [], 'query': 'Are the objects in the images in the same galaxy cluster?'}
        ]
        for query in queries:
            handler.handle(query)
        print(history.get_history())
        history.clear_history()