from config import Database

Database.initialize()
db = Database.get_db()

print("Connected collections:", db.list_collection_names())
