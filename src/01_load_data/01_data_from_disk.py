from llama_index.core import SimpleDirectoryReader
import os

# load from disk
docuemnts = SimpleDirectoryReader(os.path.abspath('../../data')).read()