import sys
import os
import time
import google.generativeai as genai
import streamlit as st
from pymongo import MongoClient

# Set up page title and sidebar
st.set_page_config(page_title="Internal AI Chatbot", page_icon="🤖")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f2f4f8;  /* Light gray background */
        }
        h1 {
            font-size: 2.5rem;  /* Increase title font size */
            color: #333;  /* Darker color for the title */
            text-align: center;
            margin-bottom: 1rem;
        }
        h2 {
            color: #4CAF50; /* Green for section headings */
            margin-top: 2rem;
        }
        .stTextInput input {
            border: 2px solid #4CAF50;  /* Green border for input */
            border-radius: 5px;
            padding: 10px;
        }
        .stButton {
            background-color: #4CAF50;  /* Green button */
            color: white;
            border-radius: 5px;
            padding: 10px 20px;  /* Add padding to the button */
        }
        .response {
            background-color: #ffffff;  /* White background for response */
            border: 1px solid #ddd;  /* Light gray border */
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px; /* Space above the response area */
        }
        .sidebar {
            background-color: #ffffff;  /* White sidebar */
            padding: 10px;
            border-radius: 5px;
        }
        .stSpinner {
            color: #4CAF50;  /* Spinner color */
        }
    </style>
""", unsafe_allow_html=True)

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

    st.sidebar.write("**Config import successful!**")
    st.sidebar.write("Google API: ", masked_api_key)
    st.sidebar.write("Mongo URI: ", masked_uri)
    st.sidebar.write("Database Name: ", DB_NAME)
    st.sidebar.write("Collection Name: ", COLLECTION_NAME)
except KeyError as e:
    st.error(f"Missing configuration for: {e}")
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

col1, col2 = st.columns([2, 3])  # Adjust the proportions as necessary

with col1:
    user_input = st.text_input(
        "Enter your query:", placeholder="What would you like to know about GitLab?")

with col2:
    if st.button("Submit"):
        if user_input:
            with st.spinner("Processing..."):
                response = query_llm_with_mongo_data(user_input)
                st.markdown(
                    f'<div class="response"><strong>Response:</strong><br>{response}</div>', unsafe_allow_html=True)
        else:
            st.error("Please enter a query.")
