# Copyright (c) Diego Colombo. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from llama_index.readers.database import DatabaseReader
import os

reader = DatabaseReader(
    scheme=os.getenv("DB_SCHEME"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    dbname=os.getenv("DB_NAME"),
)

query = "SELECT * FROM users"
documents = reader.load_data(query=query)