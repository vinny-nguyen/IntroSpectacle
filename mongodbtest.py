import os
from dotenv import load_dotenv
import pymongo

load_dotenv()

mongodb_uri = os.environ.get('MONGODB_URI')
if not mongodb_uri:
    raise ValueError("No MONGODB_URI found in environment variables.")

try:
    client = pymongo.MongoClient(mongodb_uri)
    db = client.test_database
    collection = db.test_collection
    result = collection.insert_one({'test': 'success'})
    print(f"Test document inserted with ID: {result.inserted_id}")
except Exception as e:
    print(f"An error occurred: {e}")