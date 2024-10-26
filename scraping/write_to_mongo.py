import sys
import os
from pymongo import MongoClient

# Adjust the system path to include the project root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import configuration
try:
    import config
    MONGO_URI = config.MONGO_URI
    # Mask the sensitive part of the URI for logging
    masked_uri = MONGO_URI[:23] + "****" + MONGO_URI[-16:]
    DB_NAME = config.DB_NAME
    COLLECTION_NAME = config.COLLECTION_NAME
    print("Config import successful!")
    print(f"Mongo URI: {masked_uri}")
    print(f"Database Name: {DB_NAME}")
    print(f"Collection Name: {COLLECTION_NAME}")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def connect_to_mongo():
    """Establish a connection to the MongoDB database."""
    try:
        print(f"Connecting to MongoDB with URI: {masked_uri}")
        # 30 seconds timeout
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000)
        client.admin.command('ping')  # Check connectivity
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print("Successfully connected to MongoDB.")
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)


def write_data_to_mongo(data, collection):
    """Write data (including tables) to the MongoDB collection."""
    if not data:
        print("No data to insert.")
        return

    try:
        if isinstance(data, list):
            # Insert multiple documents
            result = collection.insert_many(data)
            print(f"Data inserted with record ids: {result.inserted_ids}")
        else:
            # Insert a single document
            result = collection.insert_one(data)
            print(f"Data inserted with record id: {result.inserted_id}")
    except Exception as e:
        print(f"Error inserting data: {e}")


def main():
    # Import the web scraping function
    from webscraper import scrape_sections

    # Scrape data from the web (including tables)
    scraped_data = scrape_sections()

    # Connect to MongoDB
    collection = connect_to_mongo()

    # Write the scraped data (with tables) to MongoDB
    write_data_to_mongo(scraped_data, collection)


if __name__ == "__main__":
    main()
