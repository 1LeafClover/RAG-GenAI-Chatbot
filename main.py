
import sys
import os
import time
import google.generativeai as genai
import streamlit as st
from pymongo import MongoClient

# Set up page title and sidebar
st.set_page_config(page_title="Internal AI Chatbot", page_icon="🤖")

# Attempt to retrieve configuration from Streamlit secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    MONGO_URI = st.secrets["MONGO_URI"]
    DB_NAME = st.secrets["DB_NAME"]
    COLLECTION_NAME = st.secrets["COLLECTION_NAME"]

    # Set the GOOGLE_API_KEY as an environment variable for compatibility
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
except KeyError as e:
    st.error(f"Missing configuration for: {e}")
    sys.exit(1)

CUSTOM_PROMPT_TEMPLATE = """
# CONTEXT #
I am an internal chatbot for our organization. In the realm of internal business productivity, there is a need to enhance internal efficiency by reducing information retrieval time. Many struggle with a significant reduction in productive work hours, highlighting the need for an efficient and accurate information retrieval system.

#########

# OBJECTIVE #
Your task is to assist internal users by providing direct, detailed, and precise answers to their questions based on the internal documentation. The aim is to provide users with not only quick answers but also a thorough understanding of the topic at hand, including all relevant information from the internal documentation or knowledge base. If appropriate, expand on the information by explaining underlying concepts or offering examples to enhance clarity and comprehension.

#########

# STYLE #
Write in a clear, informative, and detailed style. Ensure responses are complete and contain all necessary information to help the user understand the topic thoroughly. Provide elaborations where needed, and avoid being overly concise.

#########

# TONE #
Maintain a neutral, professional, and helpful tone. Responses should be thorough, well-organized, and easy to follow, with clear explanations of any complex concepts.

#########

# AUDIENCE #
Internal employees seeking information from our internal documentation or knowledge base. These employees may be unfamiliar with certain topics, so detailed and clear explanations are crucial.

#########

# RESPONSE FORMAT #
Provide a comprehensive and detailed answer to the query based on the internal documentation. If the query is out of scope or not covered by the documentation, respond with:
"I'm sorry, I couldn't find the information you’re looking for. You may explore the full handbook for more information (https://handbook.gitlab.com)."

Where relevant, include examples, explanations, and references to help illustrate the key points. If information is drawn from a specific source, provide a reference with a direct link in the following format:
"Reference: [Link to source]"

#########

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
