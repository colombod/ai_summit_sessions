# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever, FUSION_MODES
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.query_engine import FLAREInstructQueryEngine, BaseQueryEngine, RetrieverQueryEngine
import os

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

llm = OpenAI(
    model="gpt-3.5-turbo-0125",
    api_key=os.getenv("OPENAI_API_KEY"),

)

Settings.llm = llm
Settings.embed_model = embed_model

# can set filters to retrieve only specific nodes
filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

print("Loading vector store...")

vector_store = DuckDBVectorStore.from_params(database_name="pg.duckdb", persist_dir=os.path.abspath("../../vector_store"), embed_dim= embed_model.dimensions if embed_model.dimensions else 1536)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

retriever = index.as_retriever(filters=filters, similarity_top_k=5)

query_rewriter = QueryFusionRetriever(
    retrievers=[retriever],
    llm=llm,
    similarity_top_k=5,
    num_queries=4,
    mode=FUSION_MODES.DIST_BASED_SCORE,
    verbose=True,
 )

query_engine = FLAREInstructQueryEngine(
    query_engine=RetrieverQueryEngine.from_args(retriever=query_rewriter,llm=llm),
    llm=llm,
    verbose=True,
)


print("Retrieving nodes using query rewriter...")
response = query_engine.query("how to .net interactive")
print(response)