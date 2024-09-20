from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

model = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    huggingfacehub_api_token="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"
)


