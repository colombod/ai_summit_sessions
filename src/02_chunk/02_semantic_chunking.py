from llama_index.core.schema import Document
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import os

documents:list[Document] = []

embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
)

Settings.embed_model = embed_model

semanticSplitterNodeParser = SemanticSplitterNodeParser(embed_model=embed_model)
semanticSplitterNodeParser.breakpoint_percentile_threshold = 95

pipeline = IngestionPipeline(
    transformations=[
        semanticSplitterNodeParser,
    ]
)

nodes = pipeline.run(documents)