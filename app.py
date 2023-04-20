from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
import requests
from bs4 import BeautifulSoup
import openai
import os

from dotenv import load_dotenv, find_dotenv

import os
import uuid
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, NotebookLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

embeddings_model = None
llm_model = None
index_name = None
index = None

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

# Add this function to verify the username and password
@auth.verify_password
def verify_password(username, password):
    if username and password:
        # You can replace these values with the ones from your environment variables or database
        correct_username = "gaii_sf_user_1"
        correct_password = "gaii_sf_password_1"
        if username == correct_username and password == correct_password:
            return True
    return False

def init():
    
    #load environment file
    load_dotenv(find_dotenv())

    global embeddings_model 
    embeddings_model = OpenAIEmbeddings(model_name="ada")
    
    global llm_model
    llm_model = OpenAI(model_name="text-davinci-003",
                       max_tokens=1000,
                       temperature=0.7,
                       n=1)

@app.route('/api/summarise/notes', methods=['POST'])
@auth.login_required  # Add this decorator to protect the route
def summarise():
    data = request.get_json()
    notes = data['notes']
    prompt = data['prompt']

    # build our prompt with the retrieved contexts included
    final_prompt = prompt + "\n\n {CONTEXT} - " + notes

    summary = llm_model(final_prompt) 
    print (summary)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    init()
    app.run(debug=True, port=8000)
