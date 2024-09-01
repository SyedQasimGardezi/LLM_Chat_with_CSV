from pandasai import Agent
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
import logging

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


def clean_data(df):
    """Perform data cleaning on the dataframe."""
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values (example: fill with mean for numerical columns)
    df = df.fillna(df.mean(numeric_only=True))

    # Convert columns to appropriate data types if needed (example: convert to datetime)
    for col in df.select_dtypes(include=['object']):
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except:
            pass

    return df


def chat_with_csv(df, prompt):
    try:
        df = clean_data(df)
        # Initialize the OpenAI language model with the API key
        llm = OpenAI(api_token=openai_api_key)

        # Initialize the Agent and set the language model
        pandas_ai = Agent(df)
        pandas_ai.llm = llm

        # Perform the chat operation with the provided dataframe and prompt
        result = pandas_ai.chat(prompt)
        return result

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"


# Set up the Streamlit app layout
st.set_page_config(layout='wide')

st.title("ChatCSV Powered by LLM")

# Initialize session state for managing the status message
if 'status_message' not in st.session_state:
    st.session_state.status_message = ""

# File uploader for the CSV file
input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        # Read the uploaded CSV file into a dataframe
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

    with col2:
        st.info("Ask Questions About the CSV Data Below")

        # Create a form for submitting queries
        with st.form(key='query_form'):
            input_text = st.text_area(
                "Enter your query", placeholder="Enter your query")
            submit_button = st.form_submit_button("Submit Query")

            if submit_button:
                if input_text:
                    # Update session state to show processing message
                    st.session_state.status_message = "Processing your query..."
                    st.write(f"### {st.session_state.status_message}")

                    # Use the chat_with_csv function to process the query
                    result = chat_with_csv(data, input_text)

                    # Update session state to clear the processing message
                    st.session_state.status_message = "Query processed."

                    st.write("### Query Result:")
                    st.write(result)
                else:
                    st.warning(
                        "Please enter a query before clicking the button.")
else:
    st.info("Please upload a CSV file to get started.")
