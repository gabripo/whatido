import os
from src.chat_handler.Query import QueryHistory, QueryCommander, QueryHandler

model_name = 'llama3.2-vision'

if __name__ == "__main__":
    history = QueryHistory()
    handler = QueryHandler()

    image_query = QueryCommander(query_history=history)
    handler.add_handler(image_query)
    
    queries = [
        {'images': [os.path.join(os.getcwd(), 'assets', 'horsehead_nebula.jpg')], 'query': 'What is in this image?'},
        {'images': [os.path.join(os.getcwd(), 'assets', 'Crab_MultiChandra_960.jpg')], 'query': 'What is in this image?'},
        {'images': [], 'query': 'Are the objects in the images in the same galaxy cluster?'}
    ]
    for query in queries:
        handler.handle(query)
    
    print(history.get_history())