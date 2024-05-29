# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.core import SimpleDirectoryReader
import os

# load from disk
docuemnts = SimpleDirectoryReader(os.path.abspath('../../data')).read()