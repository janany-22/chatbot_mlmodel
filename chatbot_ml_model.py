import streamlit as st
import pandas as pd
import numpy as np
import spacy
import sqlite3
import logging
import json
import os
import pickle
import uuid
import time
from io import StringIO
import difflib
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure data folder exists
DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Load SpaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded en_core_web_sm model successfully")
except OSError as e:
    logger.warning(f"Failed to load en_core_web_sm: {str(e)}. Using blank English model as fallback.")
    nlp = spacy.blank("en")

# Initialize components
data_store = pd.DataFrame()
available_columns = []

# SQLite database setup
DB_NAME = os.path.join(DATA_FOLDER, "chatbot_data.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            upload_time TEXT,
            file_name TEXT,
            data TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Data Preprocessing
def preprocess_data(file_content, file_name):
    start_time = time.time()
    try:
        file_ext = file_name.split(".")[-1].lower()
        if file_ext in ["csv", "txt"]:
            df = pd.read_csv(StringIO(file_content))
        elif file_ext == "json":
            df = pd.read_json(StringIO(file_content))
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(StringIO(file_content))
        elif file_ext == "pdf":
            import pdfplumber
            with pdfplumber.open(StringIO(file_content.decode('utf-8'))) as pdf:
                df = pd.DataFrame([page.extract_words() for page in pdf])
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        if df.empty:
            raise ValueError("Uploaded file is empty")

        df = df.dropna(how="all")
        df = df.fillna(df.mean(numeric_only=True).fillna(0))
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.lower().str.strip()

        logger.info(f"Preprocessed {file_name} in {time.time() - start_time:.3f} seconds")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing file {file_name}: {str(e)}")
        st.error(f"Error preprocessing file: {str(e)}")
        return None

# Store Data
def store_data(df, user_id, file_name):
    start_time = time.time()
    data_json = df.to_json()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    record_id = str(uuid.uuid4())
    c.execute(
        "INSERT INTO user_data (id, user_id, upload_time, file_name, data) VALUES (?, ?, datetime('now'), ?, ?)",
        (record_id, user_id, file_name, data_json)
    )
    conn.commit()
    conn.close()
    df.to_csv(os.path.join(DATA_FOLDER, f"{file_name}_data.csv"), index=False)
    logger.info(f"Stored data for user {user_id}, file {file_name} in {time.time() - start_time:.3f} seconds")

# Train Model (dynamic indexing for online learning)
def train_model(df):
    global data_store, available_columns
    start_time = time.time()
    try:
        data_store = df.copy()
        available_columns = [col.lower() for col in df.columns]

        # Dynamically detect identifier column (e.g., Roll No or names)
        identifier_col = None
        for col in data_store.select_dtypes(include=["object"]).columns:
            if data_store[col].astype(str).str.match(r'\w+\d+').any():  # Matches Roll No like "stu0002"
                identifier_col = col
                break
        if not identifier_col:
            for col in data_store.select_dtypes(include=["object"]).columns:
                if data_store[col].str.match(r'^[a-zA-Z\s]+$').any():  # Matches names
                    identifier_col = col
                    break
        if identifier_col:
            data_store.set_index(identifier_col, inplace=True)
            logger.info(f"Set {identifier_col} as index for row lookup")
        else:
            logger.warning("No suitable identifier column found, using original dataframe")

        # Store data_store as the "model"
        model_store = {"data_store": data_store}
        model_path = os.path.join(DATA_FOLDER, "data_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_store, f)
        logger.info(f"Trained model in {time.time() - start_time:.3f} seconds with columns: {available_columns}")
        st.success(f"Data model trained in {(time.time() - start_time) * 1000:.3f} ms with columns: {available_columns}")
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        st.error(f"Error training model: {str(e)}")

# Listener (revalidate and retrain)
def listener_check(query, user_id):
    start_time = time.time()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT data FROM user_data WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()

    for row in rows:
        df = pd.read_json(row[0])
        if "column" in query["entities"] and query["entities"]["column"] in [col.lower() for col in df.columns]:
            if df.equals(data_store):
                logger.info(f"Listener checked existing data in {time.time() - start_time:.3f} seconds, no retraining needed")
                return None
            logger.info(f"Listener detected new data, retraining model in {time.time() - start_time:.3f} seconds")
            train_model(df)
            return df
    logger.warning(f"Listener found no relevant data in {time.time() - start_time:.3f} seconds")
    return None

# Enhanced NLP Processing with SpaCy
def process_query(query):
    global data_store, available_columns
    start_time = time.time()
    doc = nlp(query.lower())
    intent = "general"
    entities = {}

    # Define keywords for intents
    intent_keywords = {
        "aggregate_average": ["average", "mean", "avg"],
        "aggregate_sum": ["sum", "total"],
        "aggregate_max": ["highest", "max", "top"],
        "aggregate_min": ["lowest", "min", "bottom"],
        "retrieve": ["what", "find", "get", "is", "tell", "show", "how", "did", "about", "for"]
    }

    # Intent detection based on keywords
    for intent_type, keywords in intent_keywords.items():
        if any(token.text in keywords for token in doc):
            intent = intent_type
            break

    # Entity extraction
    for token in doc:
        if token.text in available_columns:
            entities["column"] = token.text
            break
        matches = difflib.get_close_matches(token.text, available_columns, n=1, cutoff=0.8)
        if matches:
            entities["column"] = matches[0]
            break
        phrase = " ".join([t.text for t in doc])
        for col in available_columns:
            if col in phrase:
                entities["column"] = col
                break

    if not entities.get("column") and intent in ["aggregate_average", "aggregate_sum", "aggregate_max", "aggregate_min", "retrieve"]:
        column_context = ["mark", "score", "value", "grade", "result", "performance", "details", "about", "for", "go"]
        for token in doc:
            if token.dep_ in ["dobj", "attr", "pobj"] and any(ctx in [t.text for t in token.head.children] for ctx in column_context):
                for next_token in token.head.rights:
                    if next_token.text in available_columns:
                        entities["column"] = next_token.text
                        break
                    matches = difflib.get_close_matches(next_token.text, available_columns, n=1, cutoff=0.8)
                    if matches:
                        entities["column"] = matches[0]
                        break
                if not entities.get("column"):
                    for col in available_columns:
                        if col in query.lower():
                            entities["column"] = col
                            break

    # Extract value (e.g., Roll No or names) with improved detection
    for token in doc:
        if re.match(r'^\w+\d+$', token.text):  # Custom detection for Roll No like "stu0002"
            entities["value"] = token.text
            break
    if not entities.get("value"):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["value"] = ent.text.lower()  # Normalize to lowercase
                break
        if not entities.get("value"):
            name_tokens = []
            for token in doc:
                if token.pos_ in ["PROPN", "NOUN"] and token.text not in available_columns:
                    name_tokens.append(token.text)
            if name_tokens:
                potential_value = " ".join(name_tokens).lower()
                # Match against name column or index
                if "name" in data_store.reset_index().columns:
                    name_series = data_store.reset_index()["name"]
                    for idx, name in name_series.items():
                        if potential_value in name or name in potential_value:
                            entities["value"] = data_store.index[idx]  # Map to index (e.g., stu0005)
                            break
                if not entities.get("value"):
                    for idx_value in data_store.index:
                        if potential_value in idx_value or idx_value in potential_value:
                            entities["value"] = idx_value
                            break
                if not entities.get("value"):
                    entities["value"] = potential_value

    if not entities.get("column") and intent in ["aggregate_average", "aggregate_sum", "aggregate_max", "aggregate_min", "retrieve"]:
        logger.warning(f"No column detected in query: {query} in {time.time() - start_time:.3f} seconds")
        return {"intent": intent, "entities": entities, "raw_query": query, "status": "warning", "message": "Could not identify a valid column. Please use a column name from the dataset."}

    if intent == "retrieve" and not entities.get("value"):
        logger.warning(f"No value detected in query: {query} in {time.time() - start_time:.3f} seconds")
        return {"intent": intent, "entities": entities, "raw_query": query, "status": "warning", "message": "Please specify a value (e.g., Roll No or name) for the query."}

    logger.info(f"Processed query: {query}, Intent: {intent}, Entities: {entities} in {time.time() - start_time:.3f} seconds")
    return {"intent": intent, "entities": entities, "raw_query": query}

# Query Model
def query_model(query, user_id):
    global data_store, available_columns
    start_time = time.time()
    intent = query["intent"]
    entities = query["entities"]

    if data_store.empty:
        return {"status": "error", "message": "No data available. Please upload a file first."}

    try:
        if intent in ["aggregate_average", "aggregate_sum", "aggregate_max", "aggregate_min"]:
            col = next((col for col in data_store.columns if col.lower() == entities.get("column")), None)
            if col and col in data_store.select_dtypes(include=[np.number]).columns:
                if intent == "aggregate_average":
                    result = data_store[col].mean()
                    logger.info(f"Computed average for {col} in {time.time() - start_time:.3f} seconds")
                    return {"status": "success", "answer": f"Average {col}: {result:.2f}"}
                elif intent == "aggregate_sum":
                    result = data_store[col].sum()
                    logger.info(f"Computed sum for {col} in {time.time() - start_time:.3f} seconds")
                    return {"status": "success", "answer": f"Total {col}: {result:.2f}"}
                elif intent == "aggregate_max":
                    result = data_store[col].max()
                    logger.info(f"Computed max for {col} in {time.time() - start_time:.3f} seconds")
                    return {"status": "success", "answer": f"Highest {col}: {result:.2f}"}
                elif intent == "aggregate_min":
                    result = data_store[col].min()
                    logger.info(f"Computed min for {col} in {time.time() - start_time:.3f} seconds")
                    return {"status": "success", "answer": f"Lowest {col}: {result:.2f}"}
            else:
                raise ValueError(f"Column {entities.get('column')} not found or not numeric")
        elif intent == "retrieve":
            col = next((col for col in data_store.columns if col.lower() == entities.get("column")), None)
            value = entities.get("value", None)
            if col and value:
                matching_rows = pd.DataFrame()
                # Try indexed lookup first
                if value in data_store.index:
                    matching_rows = data_store.loc[[value], :]
                # Fallback to search name column
                if matching_rows.empty and "name" in data_store.reset_index().columns:
                    name_series = data_store.reset_index()["name"]
                    for idx, name in name_series.items():
                        if value in name or name in value:
                            matching_rows = data_store.loc[[data_store.index[idx]], :]
                            break
                # Additional fallback to search all object columns
                if matching_rows.empty:
                    for idx, row in data_store.iterrows():
                        for col_name in data_store.select_dtypes(include=["object"]).columns:
                            if value in str(row[col_name]):
                                matching_rows = data_store.loc[[idx], :]
                                break
                        if not matching_rows.empty:
                            break
                if not matching_rows.empty:
                    result = matching_rows[col].iloc[0]
                    logger.info(f"Retrieved {col} for {value} in {time.time() - start_time:.3f} seconds")
                    return {"status": "success", "answer": f"{col} for {value}: {result}"}
                else:
                    logger.warning(f"No match found for {value} in {time.time() - start_time:.3f} seconds")
                    return {"status": "error", "message": f"No data found for {value} in {col}"}
            elif not value:
                logger.warning(f"No value entity detected for retrieve intent in {time.time() - start_time:.3f} seconds")
                return {"status": "error", "message": "Please specify a value (e.g., Roll No or name) for the query."}
            else:
                raise ValueError(f"Column {entities.get('column')} not found in dataset or invalid query")

        df = listener_check(query, user_id)
        if df is not None:
            logger.info(f"Listener triggered retraining in {time.time() - start_time:.3f} seconds")
            return query_model(query, user_id)
        return {"status": "error", "message": "No relevant data found"}
    except Exception as e:
        logger.error(f"Error querying model: {str(e)} in {time.time() - start_time:.3f} seconds")
        return {"status": "error", "message": str(e)}

# Streamlit UI
st.title("Intelligent Chatbot with Online Learning")

# File Upload
uploaded_file = st.file_uploader("Upload a dataset (CSV, JSON, XLSX, TXT, PDF)", type=["csv", "json", "xlsx", "xls", "txt", "pdf"])
user_id = st.text_input("Enter User ID", value="test_user")

if uploaded_file is not None:
    start_time = time.time()
    file_content = uploaded_file.getvalue().decode("utf-8")
    df = preprocess_data(file_content, uploaded_file.name)
    if df is not None:
        store_data(df, user_id, uploaded_file.name)
        train_model(df)
        st.write(f"Dataset uploaded, stored, and model trained successfully in {(time.time() - start_time) * 1000:.3f} ms!")
        st.write(df.head())

# Query Input
query = st.text_input("Ask any question about the data (e.g., 'What is the physics mark of stu0002?', 'What is the maths mark of charvi warrior?', 'What is the highest physics mark?')")
if st.button("Submit Query") and query and not data_store.empty:
    start_time = time.time()
    processed_query = process_query(query)
    processed_query["user_id"] = user_id
    response = query_model(processed_query, user_id)
    if response.get("status") == "success":
        st.success(response["answer"])
    elif response.get("status") == "warning":
        st.warning(response["message"])
    else:
        st.error(response["message"])
    logger.info(f"Query processed in {(time.time() - start_time) * 1000:.3f} ms")