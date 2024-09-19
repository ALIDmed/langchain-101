from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    huggingfacehub_api_token="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"
)

print(llm.invoke('what is python?'))