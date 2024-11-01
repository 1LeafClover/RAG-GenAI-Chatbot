import sys
import os
import nltk
import pandas as pd
import google.generativeai as genai
import config
import time  # Import time for adding delay
from rouge_score import rouge_scorer
from pymongo import MongoClient
from main import query_llm_with_mongo_data  # Importing the chatbot function
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.corpus import wordnet  # Import WordNet for synonym expansion

# Adjust the system path to include the project root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import configuration
try:
    MONGO_URI = config.MONGO_URI
    masked_uri = MONGO_URI[:23] + "****" + \
        MONGO_URI[-16:]  # Mask sensitive part of the URI
    DB_NAME = config.DB_NAME
    COLLECTION_NAME = config.COLLECTION_NAME
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# MongoDB connection setup
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Load pre-trained Sentence-BERT model for semantic search
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight model


def fetch_all_content():
    """Fetch and concatenate all content from MongoDB collection."""
    all_content = []
    for document in collection.find():
        paragraphs = document.get('content', {}).get('p', [])
        all_content.extend(paragraphs)
    print(f"Fetched {len(all_content)} content chunks from MongoDB.")
    return all_content  # Return as a list of paragraphs


def expand_query_with_synonyms(query):
    """Expand the query using synonyms from WordNet."""
    nltk.download('wordnet')  # Ensure WordNet is available
    nltk.download('punkt')  # Ensure Punkt tokenizer models are available
    words = nltk.word_tokenize(query)
    expanded_query = set(words)  # Use a set to avoid duplicates

    for word in words:
        synonyms = wordnet.synsets(word)
        for syn in synonyms[:2]:  # Limit to the first two synsets for each word
            for lemma in syn.lemmas():
                # Add synonym only if it is not too different from the original word length
                if lemma.name() != word and abs(len(lemma.name()) - len(word)) <= 3:
                    expanded_query.add(lemma.name())  # Add synonyms to the set
                if len(expanded_query) >= 20:  # Limit the number of synonyms
                    break

    print(f"Expanded query: {expanded_query}")
    return list(expanded_query)


def retrieve_relevant_chunks_semantic(query, content_chunks, top_n=4, similarity_threshold=0.3):
    """
    Retrieve the most relevant content chunks using a combination of keyword matching, 
    Sentence-BERT embeddings, and entity-based filtering.
    """
    # Step 1: Expand query with synonyms
    query_expanded = expand_query_with_synonyms(query)

    # Step 2: Perform keyword matching to filter out irrelevant content
    keyword_filtered_chunks = [chunk for chunk in content_chunks if any(
        keyword in chunk.lower() for keyword in query_expanded)]

    print(f"Keyword filtered chunks: {keyword_filtered_chunks}")

    # Ensure there are relevant chunks after keyword filtering
    if not keyword_filtered_chunks:
        print("No relevant content found with the given query.")
        return ""

    # Step 3: Generate embeddings for the query and filtered chunks
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = model.encode(
        keyword_filtered_chunks, convert_to_tensor=True)

    # Step 4: Calculate cosine similarities between the query and all filtered chunks
    cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
    print(f"Cosine scores: {cosine_scores}")

    # Step 5: Apply a similarity threshold to filter out low-scoring chunks
    top_scores_and_indices = [(score, i) for i, score in enumerate(
        cosine_scores[0]) if score >= similarity_threshold]

    if not top_scores_and_indices:
        print("No relevant content found above the similarity threshold.")
        return ""

    # Step 6: Sort by similarity scores and pick top-N
    top_scores_and_indices.sort(reverse=True, key=lambda x: x[0])
    top_indices = [i for score, i in top_scores_and_indices[:top_n]]
    print(f"Top indices: {top_indices}")

    # Step 7: Retrieve and return the top-N most relevant chunks
    relevant_chunks = [keyword_filtered_chunks[i] for i in top_indices]
    print(f"Retrieved relevant chunks: {relevant_chunks}")

    return " ".join(relevant_chunks)  # Combine chunks into a single text


def calculate_rouge_l_precision_recall(reference, candidate):
    """Calculate ROUGE-L score, precision, and recall between reference and candidate."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    rouge_L_fmeasure = scores['rougeL'].fmeasure
    rouge_L_precision = scores['rougeL'].precision
    rouge_L_recall = scores['rougeL'].recall

    return rouge_L_fmeasure, rouge_L_precision, rouge_L_recall


def grade_scores(rouge_l):
    """Grade the performance based on ROUGE-L score with finer granularity."""
    if rouge_l > 0.7:
        return "Very High Quality"
    elif rouge_l > 0.5:
        return "High Quality"
    elif rouge_l > 0.3:
        return "Moderate Quality"
    else:
        return "Low Quality"


def run_batch_test(file_path):
    """Run batch testing with queries from an Excel file, saving results back to it."""
    df = pd.read_excel(file_path)  # Load test queries from Excel

    # Fetch all content from the database as separate chunks
    content_chunks = fetch_all_content()
    if not content_chunks:
        print("No data found in the database.")
        return

    # Prepare lists to store results
    responses, references, rouge_scores, rouge_precisions, rouge_recalls, score_grades = (
        [], [], [], [], [], [])

    # Process each query
    for index, row in df.iterrows():
        # Get the value of 'scope' and handle NaN or non-string cases
        scope = row.get('scope', '')
        if not isinstance(scope, str):
            scope = ''  # Default to an empty string if it's not a string
        scope = scope.lower()

        query = row['query']
        print(f"Processing query: {query}")

        # Get chatbot response
        response = query_llm_with_mongo_data(
            query) or "No relevant information found."
        responses.append(response)

        if scope == 'in':
            # Retrieve relevant content chunks for this query using the correct function
            relevant_chunks = retrieve_relevant_chunks_semantic(
                query, content_chunks)
            if relevant_chunks:  # Ensure there are relevant chunks
                references.append(relevant_chunks)  # Save the reference text

                # Calculate ROUGE-L score, precision, and recall between relevant chunks and response
                rouge_l, precision, recall = calculate_rouge_l_precision_recall(
                    relevant_chunks, response)
                rouge_precisions.append(f"{precision:.5f}")
                rouge_recalls.append(f"{recall:.5f}")
                rouge_scores.append(f"{rouge_l:.5f}")

                # Grade the performance based on the ROUGE-L score
                grade = grade_scores(rouge_l)
                score_grades.append(grade)
            else:
                # Append placeholders if no relevant chunks found
                references.append("N/A")
                rouge_scores.append("N/A")
                rouge_precisions.append("N/A")
                rouge_recalls.append("N/A")
                score_grades.append("N/A")

            time.sleep(30)

        elif scope == 'out':
            references.append("N/A")
            rouge_scores.append("N/A")
            rouge_precisions.append("N/A")
            rouge_recalls.append("N/A")
            score_grades.append("N/A")

        else:
            # If no scope is defined, still append placeholder values to avoid length mismatch
            references.append("N/A")
            rouge_scores.append("N/A")
            rouge_precisions.append("N/A")
            rouge_recalls.append("N/A")
            score_grades.append("N/A")

    # Ensure the lengths of the lists match the DataFrame
    while len(references) < len(df):
        references.append("N/A")
        rouge_scores.append("N/A")
        rouge_precisions.append("N/A")
        rouge_recalls.append("N/A")
        score_grades.append("N/A")

    # Add results back to the DataFrame
    df['gemini-1.5-pro-latest'] = responses
    df['reference'] = references  # Add the retrieved reference text
    df['rouge_l'] = rouge_scores
    df['precision'] = rouge_precisions
    df['recall'] = rouge_recalls
    df['score_grade'] = score_grades

    # Save updated DataFrame to the same Excel file
    df.to_excel(file_path, index=False)
    print("Batch test completed. Responses, references, rouge-l scores, precision, recall, and grades saved.")


if __name__ == "__main__":
    # Configure API key
    genai.configure(api_key=config.GOOGLE_API_KEY)

    # Path to test queries file
    file_path = r"C:\Everything\SUSS\DA\ANL488\GenAI-Chatbot-for-InternalDocumentation-and-KnowledgeBases\data\test_queries.xlsx"

    # Run the batch test
    run_batch_test(file_path)
