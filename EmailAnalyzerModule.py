import streamlit as st
import importlib.util
from streamlit_option_menu import option_menu


import Email_Summarizer
import EmailResponse
import WhoSaidWhat
import QnAwithMeetScript
import QnAwithemail
# import PDF_GPT
# import Home

# Define a function to import and run a page script dynamically
def run_page(page_name):
    spec = importlib.util.spec_from_file_location(page_name, f"pages/{page_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

# Configure Streamlit page
#st.set_page_config(
 #   page_title="TA Chatbot",
  #  page_icon="🤖",
#)

# Define page names and their corresponding display names
def app():
    st.subheader("Email Analyser")

    app = option_menu(None,['Email Summarizer', 'Generate Next Response','QnA'],
        # menu_title="Select the menu below :",
        icons=['wechat', 'filetype-pdf','megaphone'],
        default_index=0, 
        orientation='horizontal', # Set default index to 0 (Chatbot)
        styles={
            "container": {"padding": "15!important", "background-color": '#12141c',"text-align": "center"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "12px", "text-align": "center", "margin": "0px",
                         "--hover-color": "blue"},
            "nav-link-selected": {"background-color": "#FE6771","font-size": "12px"},
            "option": {"color": "black", "text-align": "right"},
        }
    )


    if app == 'Email Summarizer':
        Email_Summarizer.app()
    if app == 'Generate Next Response':
        EmailResponse.app()
    elif app == 'QnA':
        QnAwithemail.app()


if __name__ == '__main__':
  app()
