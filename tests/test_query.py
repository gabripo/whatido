import os, unittest
import _setup_test_env
from src.chat_handler import Query

class TestQueryHistory(unittest.TestCase):
    def test_history_creation(self):
        history = Query.QueryHistory()
        self.assertIsNot(history._instance, None)
        self.assertEqual(history.history, [])
        Query.QueryHistory.reset()

    def test_history_getter(self):
        history = Query.QueryHistory()
        self.assertEqual(history.history, [])
        self.assertEqual(history.get_history(), [])

        history.history = ['test']
        self.assertEqual(history.get_history(), ['test'])
        Query.QueryHistory.reset()

    def test_history_add(self):
        Query.QueryHistory.reset()
        history = Query.QueryHistory()
        self.assertEqual(history.history, [])

        history.add_message('test')
        self.assertEqual(history.history, ['test'])

    def test_history_clear(self):
        history = Query.QueryHistory()
        history.add_message('test')

        history.clear_history()
        self.assertEqual(history.history, [])
        Query.QueryHistory.reset()

class TestQueryCommander(unittest.TestCase):
    def test_commander_creation(self):
        history = Query.QueryHistory()
        commander = Query.QueryCommander(query_history=history)
        self.assertEqual(commander.query_history, history)
        Query.QueryHistory.reset()

    def test_commander_execute(self):
        pass

class TestQueryHandler(unittest.TestCase):
    def test_handler_creation(self):
        handler = Query.QueryHandler()
        self.assertEqual(handler.handlers, [])

    def test_handler_add(self):
        handler = Query.QueryHandler()
        self.assertEqual(handler.handlers, [])

        handler.add_handler('test')
        self.assertEqual(handler.handlers, ['test'])

    def test_handler_add_handle(self):
        history = Query.QueryHistory()
        commander = Query.QueryCommander(query_history=history)
        handler = Query.QueryHandler()

        handler.add_handler(commander)
        self.assertEqual(handler.handlers, [commander])
        Query.QueryHistory.reset()

    def test_handler_execute_handle_text_single(self):
        history = Query.QueryHistory()
        commander = Query.QueryCommander(query_history=history)
        handler = Query.QueryHandler()
        
        self.assertEqual(commander.query_history.history, [])

        handler.add_handler(commander)
        test_query = {'query': 'write a short sentence'}
        handler.handle(test_query)
        self.assertNotEqual(commander.query_history.history, [])
        Query.QueryHistory.reset()

    def test_handler_execute_handle_text_multiple(self):
        history = Query.QueryHistory()
        commander = Query.QueryCommander(query_history=history)
        handler = Query.QueryHandler()
        
        self.assertEqual(commander.query_history.history, [])

        handler.add_handler(commander)
        test_queries = [
            {'query': 'write a short sentence'},
            {'query': 'continue the previous sentence'}
        ]
        for test_query in test_queries:
            handler.handle(test_query)
            self.assertNotEqual(commander.query_history.history, [])
        self.assertEqual(len(commander.query_history.history), len(test_queries))
        Query.QueryHistory.reset()

    def test_handler_execute_handle_image_single(self):
        history = Query.QueryHistory()
        commander = Query.QueryCommander(query_history=history)
        handler = Query.QueryHandler()
        
        self.assertEqual(commander.query_history.history, [])

        handler.add_handler(commander)
        current_folder = os.path.dirname(__file__)
        test_query = {'images': [os.path.join(current_folder, 'assets', 'horsehead_nebula.jpg')], 'query': 'What is in this image?'}
        handler.handle(test_query)
        self.assertNotEqual(commander.query_history.history, [])
        Query.QueryHistory.reset()

    def test_handler_execute_handle_image_multiple(self):
        history = Query.QueryHistory()
        commander = Query.QueryCommander(query_history=history)
        handler = Query.QueryHandler()
        
        self.assertEqual(commander.query_history.history, [])

        handler.add_handler(commander)
        current_folder = os.path.dirname(__file__)
        test_queries = [
            {'images': [os.path.join(current_folder, 'assets', 'horsehead_nebula.jpg')], 'query': 'What is in this image?'},
            {'images': [os.path.join(current_folder, 'assets', 'Crab_MultiChandra_960.jpg')], 'query': 'What is in this image?'},
            {'images': [], 'query': 'Are the objects in the images in the same galaxy cluster?'}
        ]
        for test_query in test_queries:
            handler.handle(test_query)
            self.assertNotEqual(commander.query_history.history, [])
        self.assertEqual(len(commander.query_history.history), len(test_queries))
        Query.QueryHistory.reset()

        
if __name__ == '__main__':
    unittest.main(verbosity=2)