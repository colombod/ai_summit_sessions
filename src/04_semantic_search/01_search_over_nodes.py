# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
import os

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
Settings.embed_model = embed_model

# can set filters to retrieve only specific nodes
filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

vector_store = DuckDBVectorStore.from_local("./persist/pg.duckdb")
index = VectorStoreIndex.from_vector_store(vector_store)

retriever = index.as_retriever(filters=filters, similarity_top_k=5)

found = retriever.retrieve("semantic search")
