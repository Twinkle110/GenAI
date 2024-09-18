import streamlit as st
from io import StringIO
import os
 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
 
load_dotenv()
 
def generate_sentiment_analysis(text_data, system_message_content, vector_store):
    # Define the chat model for OpenAI (using ChatCompletion instead of OpenAI class)
    llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Initialize RetrievalQA with the chat model
    qa = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    # Generate the prompt for the task
    prompt = f"{system_message_content}\nPlease summarize email: {text_data}"
    # Get the result from the QA system
    result = qa({"query": prompt})
    return result
 
def app():
    st.write("Please input email to summarize")
 
    # Text area for input
    source_text = st.text_area("Input Text", height=200)
 
    # Submit button
    if st.button("Submit"):
        if not source_text.strip():
            st.error("Please provide an email to analyze.")
        else:
            # Convert text to string IO
            stringio = StringIO(source_text)
            string_data = stringio.read()
 
            # Text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
            )
 
            texts = text_splitter.split_text(string_data)
 
            # OpenAI embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
 
            # Vector store
            vector_store = FAISS.from_texts(texts, embeddings)
 
            # Access system message content
            system_message_content = """
            You are a highly skilled AI trained in email comprehension and summarization. I would like you to read the given email and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire email..Please follow the specified sequence each in new line with bullet points : 1) Recipients, 2) Sender, 3) Subject, and 4) Email Body 5) Next Step."
            """
 
            # Generate sentiment analysis (summary in this case)
            result = generate_sentiment_analysis(string_data, system_message_content, vector_store)
            st.write("**Output:**")
            st.code(result["result"])
 
# Call the app function to execute it
if __name__ == '__main__':
    app()
