import streamlit as st
from io import StringIO
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pyperclip
#import win32clipboard as clipboard

load_dotenv()


def work_anniversary_message(text_data, system_message_content, vector_store):
    prompt = f"{system_message_content}\nPlease draft an good email: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return qa({"query": prompt})

def app():
    content = "twinkle"
    #st.subheader("Choose the Right Words to Celebrate Your Coworkers’ Work Milestones")

    # Text input for Name
    Email = st.text_area("Enter email:", height=200)

    # Number input for Number of years worked
    #years_worked = st.number_input("Number of years worked", min_value=0, step=1)

    # Text input for Achievements
    Instruction = st.text_area("Instruction:")

    # Selector for Relationship
    # relationship_options = ["Colleague", "Manager", "Subordinates"]
    # relationship = st.selectbox("Relationship", relationship_options)

    # Selector for Tone
    tone_options = ["Formal", "Neutral", "Informal"]
    tone = st.selectbox("Tone", tone_options)

    # Submit button
    if st.button("Generate Email",key=1234):
        print("Inside if",content)
        if not Email.strip() or not tone.strip()  or not  Instruction.strip():
            print("Inside 2nd if",content)
            st.error("Please provide all the details.")
        else:
            # Convert inputs to string
            print("Inside else",content)
            string_data = f"Email: {Email}\nTone: {tone}\nInstruction: {Instruction}"

            # Text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
            )

            texts = text_splitter.split_text(string_data)

            # OpenAI embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

            # Vector store
            vector_store = FAISS.from_texts(texts, embeddings)

            # Access system message content
            # system_message_content = "You are a highly skilled AI trained in generating work anniversary message. Please read the given details like name, number of worked years, relationship, tone and draft a good work anniversary message"
            system_message_content ="You are a highly skilled AI trained in generating professional email responses. I would like you to read the given email and draft a response based on given instructions that is clear, concise, and aligns with the tone and intent of the original message. Your response should address any questions or concerns raised, provide relevant information or actions, and maintain a polite and professional tone. Please follow the specified sequence, each in a new line with bullet points: 1) Recipients, 2) Subject, 3) Email Body, 4) Closing Remarks."
            
            # Generate work anniversary message
            result = work_anniversary_message(string_data, system_message_content, vector_store)
            st.write("**Output:**")
            
            content = result["result"]
            st.write(result["result"])
            
            st.button("Copy",on_click=copy_to_clipboard(content),key=123)
        
        # pyperclip.copy(res)  # Copy text to clipboard using pyperclip
        # st.success("Message copied to clipboard!")
        
def copy_to_clipboard(res):
    pyperclip.copy(res)  # Copy text to clipboard using pyperclip
    st.success("Message copied to clipboard!")
    # print("Inside fun",res)
    # if st.button("Copy"):
    #     print("Inside Fun:")         

            # if st.button("Download Message"):
            #  message_content = otp.encode("utf-8")  # Encode for byte data
            #  st.download_button(label="Download Message", data=message_content, file_ext="txt")

# Call the app function to execute it
if __name__ == '__main__':
    app()
