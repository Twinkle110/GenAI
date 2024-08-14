import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from docx import Document
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import tempfile

# Load environment variables
load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #2b313e
}
.chat-message .avatar {
  width: 5%;
}
.chat-message .avatar img {
  max-width: 20px;
  max-height: 20px;
  border-radius: 10%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

system_message_content = "You are an intelligent AI which analyses text from documents and answers the user's questions. Please answer in as much detail as possible, so that the user does not have to revisit the document. If you don't know the answer, say that you don't know, and avoid making up things."

def get_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_text_from_doc(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    prompt_template = """
    You are an advanced Interview Analyzer. Your role is to assess job interviews by summarizing the interviewee's overall performance, evaluating their confidence level, and analyzing their sentiment throughout the interview. Provide constructive and actionable feedback, highlighting strengths and areas for improvement. Use a supportive and professional tone, focusing on verbal cues (if available).\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    llm = ChatOpenAI(
        model_name="gpt-4o"
    )
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def transcribe_audio(audio_file_path):
    """Transcribe the audio using OpenAI's Whisper model."""
    audio_file = open(audio_file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

def app():
    st.markdown(
        """
        <style>
        div[data-testid="stForm"]{
            position: fixed;
            bottom: 0;
            width: 55%;
            background-color: #2b313e;
            padding: 10px;
            z-index: 10;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("Interview Analyser:")
    file_docs = st.file_uploader("Upload your DOC, TXT, or Audio files here:", type=["docx", "txt", "wav", "mp3"], accept_multiple_files=True)
        
    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text = ""
            for file in file_docs:
                file_path = save_uploaded_file(file)
                if file.name.endswith(("wav", "mp3")):
                    raw_text += transcribe_audio(file_path)
                elif file.name.endswith("txt"):
                    raw_text += get_text_from_txt(file_path)
                elif file.name.endswith("docx"):
                    raw_text += get_text_from_doc(file_path)

            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
    
    with st.container():
        myform = st.form(key="form", clear_on_submit=True)
        user_question = myform.text_input("Ask your question: ", key='user_input', value='')

    with st.container(): 
        if myform.form_submit_button("Submit"):
            if user_question:
                handle_userinput(user_question)

if __name__ == '__main__':
    app()
