import logging
import lancedb
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import openai

EMB_MODEL_NAME = ""
DB_TABLE_NAME = ""

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable multiple retrievers
retrievers = {}

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens

def trim(text, length = 8190):
  text = ' '.join(text.split()).replace('<|endoftext|>','')
  while num_tokens_from_string(text) > length:
    text = ' '.join(text.split()[:-10])
  return text

def openai_embedding(text, key = None):
  client = openai.OpenAI(
      api_key=key,
  )
  trimmed = trim(t)
  rs = client.embeddings.create(input=[trimmed], model="text-embedding-ada-002")
  return rs.data[0].embedding

minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
gtelarge = SentenceTransformer('thenlper/gte-large')
retrievers['MiniLM'] = lambda t, key: minilm.encode(t)
retrievers['GteLarge'] = lambda t, key: gtelarge.encode(t)
retrievers['OpenAI'] = openai_embedding

# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
db = lancedb.connect(db_uri)
tables = {}
for table_name in db.table_names():
  tables[table_name] = db.open_table(table_name)
