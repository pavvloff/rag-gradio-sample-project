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
retrievers = {}
retrievers['MiniLM'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
retrievers['GteLarge'] = SentenceTransformer('thenlper/gte-large')
retrievers['OpenAI'] = None

# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
db = lancedb.connect(db_uri)
tables = {}
for table_name in db.table_names():
  tables[table_name] = db.open_table(table_name)
