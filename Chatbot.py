# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import  pyperclip

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    #st.set_page_config(
     #   page_title="TA ChatGPT",
      #  page_icon="🤖"
    #)


def app():
    init()
    st.markdown(
        """
        <style>
        div[data-testid="stForm"]{
            position: fixed;
            bottom: 0;
            width: 50%;
            background-color: #424c54;
            padding: 10px;
            z-index: 10;
        }
        </style>
        """, unsafe_allow_html=True
    )

    def message(msg, is_user=False, key=None):
     if is_user:
        avatar_url = "https://api.dicebear.com/8.x/fun-emoji/svg?seed=Simon"  # Replace with the URL of the user's avatar image
        st.markdown(f'<div class="message user-message" key="{key}"><img src="{avatar_url}" class="avatar">{msg}</div>', unsafe_allow_html=True)
     else:
        avatar_url = "https://api.dicebear.com/8.x/bottts/svg"  # Replace with the URL of the AI's avatar image
        st.markdown(f'<div class="message ai-message" key="{key}"><img src="{avatar_url}" class="avatar">{msg}</div>', unsafe_allow_html=True)


    st.markdown("""
    <style>
    .message {
        padding: 10px 10px;
        margin: 10px 0;
        border-radius: 10px;
        max-width: 100%;
    }
    .user-message {
        background-color: #2b313e; /* Adjust color as needed */
        color: white; /* Adjust text color as needed */
        text-align: left; /* Always align user messages to the left */
    }
    .ai-message {
        background-color: #475063; /* Adjust color as needed */
        color: white; /* Adjust text color as needed */
        text-align: left; /* Align AI messages to the right */
    }
    .avatar {
        width: 30px; /* Adjust avatar width as needed */
        height: 30px; /* Adjust avatar height as needed */
        margin-right: 10px; /* Adjust spacing between avatar and message */
        border-radius: 50%; /* Make avatar circular */
    }
    </style>
""", unsafe_allow_html=True)


    chat = ChatOpenAI(model="gpt-4-turbo",temperature=0)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("People Success GPT 🤖")

    # sidebar with user input
    # with st.sidebar:
    with st.container():
        myfrom=st.form(key="form",clear_on_submit=True)
        user_input = myfrom.text_area("Ask your question: ", key='user_input',value='')

    with st.container(): 
        if myfrom.form_submit_button("Submit"):
        # handle user input
            if user_input:
              st.session_state.messages.append(HumanMessage(content=user_input))
              with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
              st.session_state.messages.append(
                AIMessage(content=response.content))
              

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')
    
   

    # st.button("End", key=123)
    if len(messages) > 1:
        st.button("End", key=123)
        # st.button("End")
    # copy_to_clipboard() 

    # st.button("End", key=123)


if __name__ == '__main__':
    app()
