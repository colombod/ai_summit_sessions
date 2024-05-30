# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.core.schema import Document, BaseNode
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, JSONNodeParser,SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
import os

json_documents:list[Document] =  SimpleDirectoryReader(os.path.abspath("../../data/json"), filename_as_id=True).load_data(show_progress=True)
print(f"Loaded {len(json_documents)} json documents.")
markdown_documents:list[Document] =  SimpleDirectoryReader(os.path.abspath("../../data/markdown"), filename_as_id=True).load_data(show_progress=True)
print(f"Loaded {len(markdown_documents)} markdown documents.")

nodes:list[BaseNode] = []

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

Settings.embed_model = embed_model
semanticSplitterNodeParser = SemanticSplitterNodeParser(embed_model=embed_model)
semanticSplitterNodeParser.breakpoint_percentile_threshold = 95

markdownPipeline = IngestionPipeline(
    transformations=[
        MarkdownNodeParser(),
        semanticSplitterNodeParser,
        embed_model,
    ]
)

nodes.extend(markdownPipeline.run(documents=markdown_documents, show_progress=True))

jsonPipeline = IngestionPipeline(
    transformations=[
        JSONNodeParser(),
        embed_model,
    ]
)

nodes.extend(markdownPipeline.run(documents= markdown_documents, show_progress=True))
print(f"Created {len(nodes)} nodes.")

# prepare a duckdb vector store
vector_store = DuckDBVectorStore(database_name="pg.duckdb", persist_dir=os.path.abspath("../../vector_store"), embed_dim= embed_model.dimensions if embed_model.dimensions else 1536)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#persist data
index =  VectorStoreIndex(nodes=nodes, storage_context=storage_context)
retriever = index.as_retriever()

found = retriever.retrieve("semantic search")

