# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.core.schema import Document
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
import os

documents:list[Document] = SimpleDirectoryReader(os.path.abspath("../../data/markdown"), filename_as_id=True).load_data(show_progress=True)

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
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

nodes = pipeline.run(documents=documents, show_progress=True)

print(f"Created {len(nodes)} nodes from {len(documents)} documents.")

for node in nodes:
    print(f"---------------------------node[{node.id_}] :\n{node.get_content()}")