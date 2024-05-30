# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
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

vector_store = DuckDBVectorStore.from_params(database_name="pg.duckdb", persist_dir=os.path.abspath("../../vector_store"), embed_dim= embed_model.dimensions if embed_model.dimensions else 1536)

                                
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

retriever = index.as_retriever()

found = retriever.retrieve("polyglot notebook")

for f in found: 
    print(f.node_id, f.score)
