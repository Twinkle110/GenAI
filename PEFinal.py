import streamlit as st
import os
from io import BytesIO
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_minutes(text_data,system_message_content, vector_store):
    prompt = f"{system_message_content}\nPlease evaluate the wrap up data: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please upload a .pdf file to generate minutes of the meeting.")

    uploaded_file = st.file_uploader("Choose a .pdf file", "pdf")

    # Chat option
    # user_question = st.text_input("Ask a question about the document:")

    # Enable button only if file is uploaded
    if uploaded_file is not None:
        submit_button = st.button('Generate Output')
    else:
        submit_button = None

    if submit_button:
        # Read the uploaded PDF document
        pdf_data = BytesIO(uploaded_file.read())
        text_data = extract_text_from_pdf(pdf_data)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
        )

        texts = text_splitter.split_text(text_data)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        vector_store = FAISS.from_texts(texts, embeddings)

        # Access system message content
        system_message_content = "You are an AI assistant tasked with evaluating and summarizing annual employee performance appraisal comments (Wrap Up Report) data. User will share employee's performance data with you which will have 3 columns 1) Questions 2) Employee Comments 3) Manager Comments.Being an AI assistant your task is to evaluate every question mentioned in 1st column i.e. Question and corresponding employee comments for that question and manager comments, for example if 1st question is 1)Reflect on the key areas where you have impacted the business : What are you proud of? use employee comments and manager comments to reconcile any disparities between the manager's and employee's comments,is there any alignment between how the employee sees his/her performance and how the manager perceives it and vice versa. Identify areas where the employee excels and areas for improvement as indicated by the manager. Your responses should be accurate and precise, acknowledging any uncertainties or lack of information rather than providing misleading answers.Please provide response in different sections for all 6 questions.Try to also give a match percentage based on alignment on both employee and manager comments." 

        result = generate_minutes(text_data,system_message_content,vector_store)
        st.write("**Output:**")
        st.write(result["result"])
    else:
        st.warning("Please upload a document to proceed.")

# Call the app function to execute it
if __name__ == '__main__':
    app()
