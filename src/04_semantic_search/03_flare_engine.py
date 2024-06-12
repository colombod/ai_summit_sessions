from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever, FUSION_MODES
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.query_engine import FLAREInstructQueryEngine, RetrieverQueryEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # take environment variables from .env.

# Initialize embedding model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize LLM
llm = OpenAI(
    model="gpt-3.5-turbo-0125",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Set global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Set filters to retrieve only specific nodes
filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

print("Loading vector store...")

# Ensure correct dimension is used
embed_dim = getattr(embed_model, 'dimensions', None)
if embed_dim is None:
    embed_dim = 1536  # Default to 1536 if dimensions attribute is missing
print(f"Using embedding dimension: {embed_dim}")

# Initialize vector store
vector_store = DuckDBVectorStore.from_params(
    database_name="pg.duckdb",
    persist_dir=os.path.abspath("../../vector_store"),
    embed_dim=embed_dim
)

# Debugging: Check if vector store is loaded
if vector_store:
    print("Vector store loaded successfully.")
else:
    print("Failed to load vector store.")

# Initialize index
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Debugging: Check if index is created
if index:
    print("Index created successfully.")
else:
    print("Failed to create index.")

# Initialize retriever
retriever = index.as_retriever(similarity_top_k=5)

# Debugging: Check if retriever is created
if retriever:
    print("Retriever created successfully.")
else:
    print("Failed to create retriever.")

# Initialize query rewriter
query_rewriter = QueryFusionRetriever(
    retrievers=[retriever],
    llm=llm,
    similarity_top_k=5,
    num_queries=4,
    mode=FUSION_MODES.DIST_BASED_SCORE,
    verbose=True,
)

# Debugging: Check if query rewriter is created
if query_rewriter:
    print("Query rewriter created successfully.")
else:
    print("Failed to create query rewriter.")

# Initialize query engine
query_engine = FLAREInstructQueryEngine(
    query_engine=RetrieverQueryEngine.from_args(retriever=query_rewriter,llm=llm),
    llm=llm,
    verbose=True,
)

# Debugging: Check if query engine is created
if query_engine:
    print("Query engine created successfully.")
else:
    print("Failed to create query engine.")

print("Retrieving nodes using query rewriter...")

# Perform the query
response = query_engine.query("how to use .NET interactive")
print(response)
