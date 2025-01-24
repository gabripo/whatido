import asyncio
from .Database import Database
from ..models_interfaces.Llama import LlamaTextQuery

DEFAULT_QUERIES = ['generate an e-mail']
DEFAULT_NUM_EMAILS_TO_GEN = 10
DEFAULT_MAX_CONCURRENT_REQUESTS = 10

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

    def create_folder(self) -> None:
        return super().create_folder()

    def build(self, queries: list[dict] = DEFAULT_QUERIES, num_emails_to_gen: int = DEFAULT_NUM_EMAILS_TO_GEN):
        self.generated_data = asyncio.run(self._generate_emails(queries, num_emails_to_gen))

    async def _generate_emails(self, queries: list[dict] = DEFAULT_QUERIES, num_emails_to_gen: int = DEFAULT_NUM_EMAILS_TO_GEN):
        tasks = [self._generate_n_emails(query, num_emails_to_gen) for query in queries]
        generated_emails = await asyncio.gather(*tasks)
        return generated_emails

    async def _generate_n_emails(self, query: dict, num_emails_to_gen: int = 1):
        query_text = query['query']
        scores_dict = query['scores'].scores
        emails = [await self._llama_text_call_limited(query_text, call_number) for call_number in range(num_emails_to_gen)]
        return {'emails': emails, 'score': scores_dict}

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
    
    def print(self):
        return super().print()
    
    def set_max_concurrent_requests(self, max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)