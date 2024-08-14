import streamlit as st
import os
from io import BytesIO
import docx2txt  # Import the docx2txt library

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pyperclip
import time

load_dotenv()

def generate_minutes(text_data, user_question, system_message_content, vector_store):
    prompt = f"{system_message_content}\nUser Question: {user_question}\nPlease answer the user question: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4o"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
    )
    return qa({"query": prompt})

def app():

    st.markdown(
        """
        <style>
        div[data-testid="stForm"]{
            position: fixed;
            bottom: 0;
            width: 50%;
            background-color: #2b313e;
            padding: 10px;
            height: 195px;
            z-index: 10;
        }
        </style>
        """, unsafe_allow_html=True
    )

    session_state = st.session_state
    if 'content' not in session_state:
        session_state.content = ""

    st.write("Please enter the email you want to chat with:")

    text_data = st.text_area("Enter text here:")

    with st.form("Question", clear_on_submit=True):
        user_question = st.text_area("Ask a question about the document:")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if text_data.strip():  # Check if the text is not empty
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=20,
                    length_function=len,
                )

                texts = text_splitter.split_text(text_data)

                embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

                vector_store = FAISS.from_texts(texts, embeddings)

                system_message_content = "As an AI assistant, your task is to provide responses to user questions."

                result = generate_minutes(text_data, user_question, system_message_content, vector_store)
                session_state.content = result["result"]
            else:
                st.error("Please enter some text before submitting.")

    if len(session_state.content) > 1:
        st.write("**Output:**")
        st.write(session_state.content)

        if st.button("End", key=123):
            copy_to_clipboard(session_state.content)

def copy_to_clipboard(res):
    pyperclip.copy(res)

if __name__ == '__main__':
    app()
