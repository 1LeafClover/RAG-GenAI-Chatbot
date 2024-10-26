import sys
import os
import time
import google.generativeai as genai
import streamlit as st
from pymongo import MongoClient

# Set up page title and sidebar
st.set_page_config(page_title="Internal AI Chatbot")

# Adjust the system path to include the project root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import configuration
try:
    import config
    GOOGLE_API_KEY = config.GOOGLE_API_KEY
    MONGO_URI = config.MONGO_URI
    DB_NAME = config.DB_NAME
    COLLECTION_NAME = config.COLLECTION_NAME

    # Set the GOOGLE_API_KEY as an environment variable
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    # Initialize genai with the API key
    genai.configure(api_key=GOOGLE_API_KEY)

    # Mask the sensitive part of the API and URI
    masked_api_key = GOOGLE_API_KEY[:2] + "****" + GOOGLE_API_KEY[-4:]
    masked_uri = MONGO_URI[:23] + "****" + MONGO_URI[-16:]

    st.sidebar.write("Config import successful!")
    st.sidebar.write("Google API: ", masked_api_key)
    st.sidebar.write("Mongo URI: ", masked_uri)
    st.sidebar.write("Database Name: ", DB_NAME)
    st.sidebar.write("Collection Name: ", COLLECTION_NAME)
except ModuleNotFoundError as e:
    st.error(f"Import failed: {e}")
    sys.exit(1)

CUSTOM_PROMPT_TEMPLATE = """
# CONTEXT #
I am an internal chatbot for our organization, designed to provide detailed and accurate responses based on our internal documentation. My purpose is to support employees by retrieving and explaining information from our internal knowledge base to improve efficiency.

# USER QUERY #
{user_query}

# CONTEXT FROM MONGODB #
{mongo_data}
"""


def connect_to_mongo(retries=5, delay=5):
    """Establish a connection to the MongoDB database with retry logic."""
    attempt = 0
    while attempt < retries:
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=60000)
            client.admin.command('ping')  # Check connectivity
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            return collection
        except Exception as e:
            st.error(f"Error connecting to MongoDB: {e}")
            attempt += 1
            if attempt < retries:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                sys.exit(1)


def fetch_data_from_mongo(query=None):
    """Fetch data from MongoDB collection."""
    collection = connect_to_mongo()
    try:
        if query is None:
            query = {}  # Fetch all documents if no query is provided
        data = collection.find(query)
        documents = list(data)
        return documents
    except Exception as e:
        st.error(f"Error fetching data from MongoDB: {e}")
        return []


def query_llm_with_mongo_data(user_prompt):
    """Query the LLM using data fetched from MongoDB."""
    mongo_data = fetch_data_from_mongo()

    # Format the custom prompt by inserting the user query and MongoDB data
    formatted_data = ""
    for doc in mongo_data:
        # Add each document's fields in a readable way
        formatted_data += "\n".join([f"{key}: {value}" for key,
                                    value in doc.items() if key != 'source_link'])
        formatted_data += "\n\n"

    if not formatted_data:
        formatted_data = "No relevant data found in the internal documentation."

    custom_prompt = CUSTOM_PROMPT_TEMPLATE.format(
        user_query=user_prompt, mongo_data=formatted_data)

    try:
        # Use the gemini model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(custom_prompt)

        # Extract and clean the response content
        response_text = response._result.candidates[0].content.parts[0].text.strip(
        )
        return response_text
    except Exception as e:
        st.error(f"Error querying the LLM: {e}")
        return "I'm sorry, there was an error processing your request."


# Streamlit UI Components
st.title("GitLab Handbook IDKB Chatbot")

user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_input:
        with st.spinner("Processing..."):
            response = query_llm_with_mongo_data(user_input)
            st.write(response)
    else:
        st.error("Please enter a query.")
