import asyncio, os, json, re
from .Database import Database
from ..models_interfaces.ModelFactory import ModelFactory
from ..datatypes.Score import Score
from ..datatypes.Response import EmailResponse

DEFAULT_QUERIES = ['generate an e-mail']
DEFAULT_NUM_EMAILS_TO_GEN_PER_QUERY = 10
DEFAULT_MAX_CONCURRENT_REQUESTS = 10

class QueryScorePair():
    def __init__(self, query_text: str, scores: list):
        self.query_text = query_text
        self.scores = Score(scores)

class DatabaseEmails(Database):
    def __init__(self, name: str, path: str = "database"):
        self.database_name = name
        self.database_path = path
        self.num_emails = self.count_emails()
        self.generation_options = {
            'llamakwargs': {
                'format': EmailResponse.model_json_schema()
                },
            'include_history': False,
            'add_to_history': False
        }
        self.semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENT_REQUESTS)
        self.queries = DEFAULT_QUERIES
        self.num_emails_to_gen_per_query = DEFAULT_NUM_EMAILS_TO_GEN_PER_QUERY

    def create_folder(self) -> None:
        return super().create_folder()
    
    def get_database_abspath(self):
        return super().get_database_abspath()

    def build(self, queries: list[QueryScorePair] = DEFAULT_QUERIES, num_emails_to_gen: int = DEFAULT_NUM_EMAILS_TO_GEN_PER_QUERY):
        self.queries = queries
        self.num_emails_to_gen_per_query = num_emails_to_gen
        self.generated_data = asyncio.run(self._generate_emails())

    async def _generate_emails(self):
        tasks = [self._generate_n_emails(query) for query in self.queries]
        generated_emails = await asyncio.gather(*tasks)
        return generated_emails

    async def _generate_n_emails(self, query: QueryScorePair):
        query_text = query.query_text
        scores_dict = query.scores.scores
        emails = [
            {
            'email': await self._llama_text_call_limited(query_text, call_number),
            'score': scores_dict
            }
            for call_number in range(self.num_emails_to_gen_per_query)
            ]
        if self.store_while_generating:
            async with self.file_lock:
                for emails_dict in emails:
                    self.store_single_entry(emails_dict)
        return emails

    async def _llama_text_call_limited(self, query_text: str, call_number: int = 0):
        async with self.semaphore:
            response = await self._llama_text_call(query_text, call_number)
            try:
                return EmailResponse.model_validate_json(response)
            except:
                print(f"Fallback for response: {response} ...")
                return {'email_subject': '', 'email_text': f"{response}"}

    async def _llama_text_call(self, query_text: str, call_number: int = 0):
        factory_obj = ModelFactory.create_model_factory()
        query_obj = factory_obj.create_text_query()
        print(f"Generating for query \"{query_text}\", call {call_number} ...\n")
        kwargs = {
            'query_text': query_text,
            **self.generation_options
        }
        response = await asyncio.to_thread(
            query_obj.query,
            **kwargs
            )
        print(f"Response for query \"{query_text}\", call {call_number} has been generated!\n")
        self.num_emails += 1
        return self._format_response(response['content'])
    
    @classmethod
    def _format_response(self, response: str):
        return re.sub(
            r'<[^>]+>',
            '',
            response.encode('utf-8', errors='replace').decode('utf-8')
            )
    
    def store(self):
        return super().store()
    
    def store_single_entry(self, entry):
        return super().store_single_entry(entry)
    
    def clear(self):
        return super().clear()
    
    def print(self):
        super().print()
        print(f"E-mails in the dataset: {self.num_emails}\n")
    
    def set_max_concurrent_requests(self, max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    def count_emails(self):
        database_path = os.path.abspath(os.path.join(self.database_path, self.database_name + '.json'))
        if not self.is_json_file_empty(database_path) and self.is_json_readable(database_path):
            with open(database_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            return len(data)
        return 0