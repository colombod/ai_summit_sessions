# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
import os

documents:list[Document] = SimpleDirectoryReader(os.path.abspath("../../data/pdf"), filename_as_id=True).load_data(show_progress=True)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=100,
            chunk_overlap=20
            )
    ]
)

nodes = pipeline.run(documents=documents, show_progress=True)
print(f"Created {len(nodes)} nodes from {len(documents)} documents.")

for node in nodes:
    print(f"---------------------------node[{node.id_}] :\n{node.get_content()}")
