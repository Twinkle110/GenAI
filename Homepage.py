import streamlit as st
import importlib.util
from streamlit_option_menu import option_menu

import Chatbot
import TA_Picture_GPT
import PDF_GPT
import Home
import Email_Summarizer
import EmailAnalyzerModule
import QnAwithemail
import EmailResponse
import Meeting
import wa
import ChatwithDocuments
#import PEFinal
#import FinalInterview


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
def run():
    with st.sidebar:
        app = option_menu(
            menu_title='People Success AI Bot',
            options=['Home','Chatbot','Meeting Analyser', 'Picture GPT','Txt/Doc GPT', 'PDF GPT','Email Summarizer','Work Anniversary','Performance Evaluation','Interview Analyser'],
            icons=['house-fill','wechat','journal-check','images','filetype-txt','filetype-txt','envelope-at-fill','cake','person-workspace','person-workspace'],
            menu_icon='chat-text-fill',
            default_index=0,  # Set default index to 0 (Chatbot)
            styles={
                "container": {"padding": "5!important", "background-color": 'grey'},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {"color": "white", "font-size": "17px", "text-align": "left", "margin": "0px",
                             "--hover-color": "#9B8082"},
                "nav-link-selected": {"background-color": "#FE6771"},
            }
        )

    if app == 'Home':
        Home.app()
    if app == 'Meeting Analyser':
        Meeting.app()
    if app == 'Email Summarizer':
       EmailAnalyzerModule.app()
    if app == 'Chatbot':
        Chatbot.app()
    if app == 'Work Anniversary':
       wa.app()
    if app == 'Picture GPT':
        TA_Picture_GPT.app()
    if app == 'Txt/Doc GPT':
        ChatwithDocuments.app()
    #if app == 'Performance Evaluation':
     #   PEFinal.app()
    #if app == 'Interview Analyser':
     #   FinalInterview.app()
    elif app == 'PDF GPT':
        PDF_GPT.app()

run()
              
