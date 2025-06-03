import hnswlib
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm

# Configurazione
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "arxiv_db"
COLLECTION = "papers"
DIM = 384
INDEX_PATH = "hnsw_papers.index"
IDS_PATH = "paper_ids.npy"
BATCH_SIZE = 50_000

# Connessione a MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION]

# Cursor su tutti i paper che hanno embedding
cursor = collection.find({"embedding": {"$exists": True}}, batch_size=1000)

# Prepara oggetti
index = None
ids = []
batch_data = []
batch_ids = []
counter = 0

print("Scanning papers and building index...")

for doc in tqdm(cursor, desc="Validating + Indexing"):
    emb = doc.get("embedding")
    if isinstance(emb, list) and len(emb) == DIM:
        if index is None:
            index = hnswlib.Index(space="cosine", dim=DIM)
            count_with_embedding = collection.count_documents({"embedding": {"$exists": True}})
            index.init_index(max_elements=count_with_embedding + 1000, ef_construction=200, M=16)
            index.set_ef(50)

        batch_data.append(np.array(emb, dtype=np.float32))
        batch_ids.append(doc["id"])

        if len(batch_data) >= BATCH_SIZE:
            vecs = np.vstack(batch_data)
            index.add_items(vecs, np.arange(counter, counter + len(vecs)))
            ids.extend(batch_ids)
            counter += len(vecs)
            batch_data, batch_ids = [], []

# Ultimi elementi
if batch_data:
    vecs = np.vstack(batch_data)
    index.add_items(vecs, np.arange(counter, counter + len(vecs)))
    ids.extend(batch_ids)

# Salva
index.save_index(INDEX_PATH)
np.save(IDS_PATH, np.array(ids))

print("Index saved:", INDEX_PATH)
print("IDs saved:", IDS_PATH)
