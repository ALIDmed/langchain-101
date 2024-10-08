import os
import uuid

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")


if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"the file {file_path} doesn't exist"
    )

loader = TextLoader(file_path, autodetect_encoding=True)
documents = loader.load()

embeddings = OllamaEmbeddings(model="llama3.1")

def create_vectore_store(docs, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print(f"Creating vector store {store_name}...")

        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        db = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embedding=embeddings,
            persist_directory=persistent_dir
        )
        print(f"Finished Creating vector store {store_name}...")
    else:
        print(f"vectore store {store_name} already exists")

# 1. Character-based Splitting
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
print("### Using character-based splitting ###")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vectore_store(char_docs, "chroma_db_char")

# 2. Sentence-based Splitting
# Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
print("### Using sentence-based splitting ###")
sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sentence_docs = sentence_splitter.split_documents(documents)
create_vectore_store(sentence_docs, "chroma_db_sentence")

# 3. Token-based Splitting
# Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# Useful for transformer models with strict token limits.
print("### Using token-based splitting ###")
token_splitter = TokenTextSplitter(chunk_size=1000)
token_docs = token_splitter.split_documents(documents)
create_vectore_store(token_docs, "chroma_db_token")

# 4. Recursive Character-based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("### Using recursive character-based splitting ###")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vectore_store(rec_char_docs, "chroma_db_rec_char")

# 5. Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("### Using Custom splitting ###")
class CustomTextSplitter(TextSplitter):
    def split_text(self, text: str):
        return text.split("\n\n") # Example: split by paragraphs

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vectore_store(custom_docs, "chroma_db_custom")

def query_vectorestore(store_name, query):
    persistent_dir = os.path.join(db_dir, store_name)

    if os.path.exists(persistent_dir):
        db = Chroma(
            persist_directory=persistent_dir,
            embedding_function=embeddings
        )

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k':3, "score_threshold":0.3}
        )

        relevant_docs = retriever.invoke(query)

        print(f"--- Relevant Documents for {store_name}---")
        for i, doc in enumerate(relevant_docs):
            print(f"Document{i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"The vectore store {store_name} does not exist.")

query = "why romeo wanted to hide?"
query_vectorestore("chroma_db_rec_char", query)