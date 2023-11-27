import logging
import lancedb
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMB_MODEL_NAME = ""
DB_TABLE_NAME = ""

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable multiple retrievers
retrievers = {}

def openai_embedding(text, key = None):
  rs = client.embeddings.create(input=[text], model="text-embedding-ada-002")
  return rs.data[0].embedding

minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
gtelarge = SentenceTransformer('thenlper/gte-large')
retrievers['MiniLM'] = lambda t: minilm.encode(t)
retrievers['GteLarge'] = lambda t: gtelarge.encode(t)
retrievers['OpenAI'] = openai_embedding

# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
db = lancedb.connect(db_uri)
tables = {}
for table_name in db.table_names():
  tables[table_name] = db.open_table(table_name)
