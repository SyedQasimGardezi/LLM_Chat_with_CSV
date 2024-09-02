from pandasai import Agent
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
import logging

# Set the page configuration as the first Streamlit command
st.set_page_config(layout='wide')

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


def clean_data(df, numeric_strategy, categorical_strategy, remove_duplicates, remove_null_rows, rename_columns, use_first_row_as_header):
    """Perform data cleaning on the dataframe based on user preferences."""
    if use_first_row_as_header:
        # Set the first row as the header and remove it from data
        df.columns = df.iloc[0]
        df = df[1:]

    if rename_columns:
        # Remove extra spaces and empty names
        rename_columns = [name.strip()
                          for name in rename_columns if name.strip()]
        if len(rename_columns) == len(df.columns):
            df.columns = rename_columns
        else:
            logging.warning(
                "Number of new column names does not match number of columns in the DataFrame.")

    if remove_duplicates:
        df = df.drop_duplicates()

    if remove_null_rows:
        df = df.dropna()

    # Handle numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if numeric_strategy == 'Mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif numeric_strategy == 'Median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif numeric_strategy == 'Zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if categorical_strategy == 'Mode':
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
        elif categorical_strategy == 'Fill with "Unknown"':
            df[col].fillna('Unknown', inplace=True)

    # Convert columns to appropriate data types if needed
    for col in categorical_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except:
            pass

    return df


def chat_with_csv(df, prompt, numeric_strategy, categorical_strategy):
    try:
        if df is None or df.empty:
            raise ValueError("The DataFrame is empty or None.")

        # Clean the data according to the user's preferences
        df = clean_data(df, numeric_strategy, categorical_strategy,
                        st.session_state.remove_duplicates,
                        st.session_state.remove_null_rows,
                        st.session_state.rename_columns,
                        st.session_state.use_first_row_as_header)

        # Initialize the OpenAI language model with the API key
        llm = OpenAI(api_token=openai_api_key)

        # Initialize the Agent and set the language model
        pandas_ai = Agent(df)
        pandas_ai.llm = llm

        # Perform the chat operation with the provided dataframe and prompt
        result = pandas_ai.chat(prompt)

        # Return both the result and the cleaned dataframe
        return result, df

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An error occurred: {e}", df


st.title("ChatCSV Powered by LLM")

# Initialize session state for managing the status message and storing the dataframe
if 'status_message' not in st.session_state:
    st.session_state.status_message = ""
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'numeric_strategy' not in st.session_state:
    st.session_state.numeric_strategy = 'None'
if 'categorical_strategy' not in st.session_state:
    st.session_state.categorical_strategy = 'None'
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'remove_duplicates' not in st.session_state:
    st.session_state.remove_duplicates = False
if 'remove_null_rows' not in st.session_state:
    st.session_state.remove_null_rows = False
if 'rename_columns' not in st.session_state:
    st.session_state.rename_columns = []
if 'use_first_row_as_header' not in st.session_state:
    st.session_state.use_first_row_as_header = False

# File uploader for the CSV file
input_csv = st.file_uploader("Upload your CSV file", type=['csv', 'txt'])

if input_csv is not None:
    st.session_state.data = pd.read_csv(input_csv)
    st.session_state.preprocessed = False  # Reset preprocessing flag

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.data is not None:
            st.subheader("Original Dataset")
            st.dataframe(st.session_state.data, use_container_width=True)

        if st.session_state.preprocessed and st.session_state.cleaned_data is not None:
            st.subheader("Cleaned Dataset")
            st.dataframe(st.session_state.cleaned_data,
                         use_container_width=True)
    with col2:
        # Section for column renaming
        with st.expander("Rename Columns"):
            st.session_state.use_first_row_as_header = st.checkbox(
                "Use first row as column headers")
            rename_columns_input = st.text_area(
                "Enter new column names separated by commas (if any)",
                value=', '.join(st.session_state.rename_columns)
            )
            st.session_state.rename_columns = [
                name.strip() for name in rename_columns_input.split(',') if name.strip()]

        # Section for data cleaning options
        if st.session_state.data is not None:
            with st.expander("Data Cleaning Options"):
                st.session_state.numeric_strategy = st.selectbox(
                    "Choose a strategy to handle missing values in numerical columns",
                    ["None", "Mean", "Median", "Zero"]
                )
                st.session_state.categorical_strategy = st.selectbox(
                    "Choose a strategy to handle missing values in categorical columns",
                    ["None", "Mode", "Fill with 'Unknown'"]
                )
                st.session_state.remove_duplicates = st.checkbox(
                    "Remove duplicates")
                st.session_state.remove_null_rows = st.checkbox(
                    "Remove rows with null values")

                # Button to apply preprocessing without running the prompt
                apply_preprocessing = st.button("Apply Preprocessing")

                if apply_preprocessing:
                    # Clear previous DataFrame display
                    col1.empty()

                    # Apply preprocessing
                    st.session_state.cleaned_data = clean_data(
                        st.session_state.data,
                        st.session_state.numeric_strategy,
                        st.session_state.categorical_strategy,
                        st.session_state.remove_duplicates,
                        st.session_state.remove_null_rows,
                        st.session_state.rename_columns,
                        st.session_state.use_first_row_as_header
                    )
                    st.session_state.preprocessed = True
                    st.success("Data preprocessing applied successfully.")

                    with col1:
                        st.dataframe(st.session_state.cleaned_data,
                                     use_container_width=True)

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
                    data_to_use = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
                    if data_to_use is not None:
                        result, cleaned_df = chat_with_csv(
                            data_to_use, input_text, st.session_state.numeric_strategy, st.session_state.categorical_strategy
                        )

                        # Replace the previous DataFrame with the cleaned DataFrame in session state
                        st.session_state.cleaned_data = cleaned_df

                        # Update session state to clear the processing message
                        st.session_state.status_message = "Query processed."

                        st.write("### Query Result:")
                        st.write(result)
                else:
                    st.warning("Please enter a query.")
else:
    st.info("Please upload a CSV file to get started.")
