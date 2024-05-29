from llama_index.core.schema import Document, BaseNode
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser, MarkdownNodeParser, JSONNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.duckdb import DuckDBVectorStore

import os

json_documents:list[Document] = []
markdown_documents:list[Document] = []
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

nodes.extend(markdownPipeline.run(markdown_documents))

jsonPipeline = IngestionPipeline(
    transformations=[
        JSONNodeParser(),
        semanticSplitterNodeParser,
        embed_model,
    ]
)

nodes.extend(jsonPipeline.run(json_documents))

# prepare a duckdb vector store
vector_store = DuckDBVectorStore(persist_dir=os.path.abspath("../../vector_store"))
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#persist data
retriever = VectorStoreIndex.from_documents(nodes, storage_context=storage_context, embed_model=embed_model).as_retriever()

found = retriever.retrieve("semantic search")

