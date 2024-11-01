from dotenv import load_dotenv
import os

# Specify the correct path and filename for the env file
env_path = os.path.join(os.path.dirname(__file__), 'config.env')
load_dotenv(dotenv_path=env_path)

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
