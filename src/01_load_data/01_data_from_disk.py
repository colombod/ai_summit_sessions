# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.core import SimpleDirectoryReader
import os

# load from disk
docuemnts = SimpleDirectoryReader(os.path.abspath('../../data'), recursive=True, filename_as_id=True).load_data(show_progress=True)

for doc in docuemnts:
    print(doc.id_)