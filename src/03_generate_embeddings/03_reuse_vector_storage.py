from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
Settings.embed_model = embed_model

vector_store = DuckDBVectorStore.from_local("./persist/pg.duckdb")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

retriever = index.as_retriever()


