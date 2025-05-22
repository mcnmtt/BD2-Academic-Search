import bcrypt
from pymongo import MongoClient

client = MongoClient("mongodb://172.17.224.1:27017")
db = client["arxiv_db"]

# credenziali
username = "admin"
plaintext_password = "admin1234"

# hash password
hashed = bcrypt.hashpw(plaintext_password.encode('utf-8'), bcrypt.gensalt())

# salva utente nel DB
db["users"].insert_one({"username": username, "password": hashed})
