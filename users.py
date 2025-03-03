from pymongo import MongoClient
import bcrypt
import os
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["student_attention"]
users_collection = db["users"]

def add_user(username, password):
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed_pw})
    print(f"User {username} added successfully!")

# Add users manually
add_user("admin", "hariduvva")

