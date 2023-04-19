from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import openai
import os

from dotenv import load_dotenv, find_dotenv
import git as git_module

import os
import uuid
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.llms import OpenAI
from git import Repo
from langchain.document_loaders import DirectoryLoader, NotebookLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

embeddings_model = None
llm_model = None
index_name = None
index = None

app = Flask(__name__)
CORS(app)

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

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    
    global index_name
    index_name = 'sfindex'
    
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        
        print ('Index not present, creating')
        
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=1024
        )
        
    # connect to index
    global index
    index = pinecone.Index(index_name)
    print ('Index loaded')

@app.route('/api/summarise/notes', methods=['POST'])
def summarise():
    data = request.get_json()
    notes = data.get('notes')

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "You are a sales professional. Summarise the minutes of meeting passed in the context below. Stick to the facts mentioned in the context.\n\n"+
        "Context:\n"
    )

    prompt = prompt_start.join(notes)
    print (prompt)

    summary = llm_model(prompt) 
    print (summary)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    # init()
    app.run(debug=False)
