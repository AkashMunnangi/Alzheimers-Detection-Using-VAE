import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME")

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 16777216))

    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

    @staticmethod
    def allowed_file(filename):
        return "." in filename and \
               filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


class Database:
    client = None
    db = None

    @staticmethod
    def initialize():
        Database.client = MongoClient(Config.MONGODB_URI)
        Database.db = Database.client[Config.DATABASE_NAME]

    @staticmethod
    def get_db():
        return Database.db
