import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
#from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage,
)



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

system_message_content = "You are an intelligent AI which analyses text from documents and answers the user's questions. Please answer in as much detail as possible, so that the user does not have to revisit the document. If you don't know the answer, say that you don't know, and avoid making up things.You are designed to provide assistance without bias based on religion, ethnicity, or caste. You do not assess emotions. Your goal is to offer respectful and impartial support."

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):

    prompt_template = """
    You are an intelligent AI which analyses text from pdf, pdfs and answers the user's questions.Please answer in as much detail as possible, so that the user does not have to 
    revisit the document.If you don't know the answer, say that you don't know, and avoid making up things.Please be very accurate with what user is required.\n\n
    Context:\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    llm = ChatOpenAI(
            #model_name="gpt-3.5-turbo",
            model_name="gpt-4o"
            # model_name="gpt-4-turbo-2024-04-09"  
    )
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    
    memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
    
    qgc = LLMChain(llm=llm,prompt=PROMPT)
    
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
   

def app():
    load_dotenv()

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
    # new()

    # st.set_page_config(page_title="Chat with multiple PDFs",
    #                    page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # st.header("Chat with your document :books:")

    # with st.sidebar:
    st.subheader("Chat with your document :books:")
    pdf_docs = st.file_uploader("Upload your PDFs here'", type=["pdf"],accept_multiple_files=True)
        
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
    
    with st.container():
        myfrom=st.form(key="form",clear_on_submit=True)
        user_question = myfrom.text_input("Ask your question: ", key='user_input',value='')

    with st.container(): 
        if myfrom.form_submit_button("Submit"):
            if user_question:
                handle_userinput(user_question)

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            
    
            
     
if __name__ == '__main__':
   app()
