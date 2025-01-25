import asyncio
from .Database import Database
from ..models_interfaces.Llama import LlamaTextQuery
from .Score import Score

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
        self.num_emails = 0
        self.generation_options = {
            'include_history': False,
            'consider_history': False
        }
        self.semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENT_REQUESTS)
        self.queries = DEFAULT_QUERIES
        self.num_emails_to_gen_per_query = DEFAULT_NUM_EMAILS_TO_GEN_PER_QUERY

    def create_folder(self) -> None:
        return super().create_folder()

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
        emails = [{'email': await self._llama_text_call_limited(query_text, call_number), 'score': scores_dict} for call_number in range(self.num_emails_to_gen_per_query)]
        if self.store_while_generating:
            async with self.file_lock:
                for emails_dict in emails:
                    self.store_single_entry(emails_dict)
        return emails

    async def _llama_text_call_limited(self, query_text: str, call_number: int = 0):
        async with self.semaphore:
            return await self._llama_text_call(query_text, call_number)

    async def _llama_text_call(self, query_text: str, call_number: int = 0):
        query_obj = LlamaTextQuery()
        print(f"Generating for query \"{query_text}\", call {call_number} ...\n")
        response = await asyncio.to_thread(
            query_obj.query,
            query_text,
            self.generation_options['include_history'],
            self.generation_options['consider_history']
            )
        print(f"Response for query \"{query_text}\", call {call_number} has been generated!\n")
        self.num_emails += 1
        return response['content']
    
    def store(self):
        return super().store()
    
    def store_single_entry(self, entry):
        return super().store_single_entry(entry)
    
    def clear(self):
        return super().clear()
    
    def print(self):
        return super().print()
    
    def set_max_concurrent_requests(self, max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)