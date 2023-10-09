# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 02:06:48 2023

@author: COMPUTER
"""

import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer
from apikey import apikeys_huggingface

#set huggingface API token

os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikeys_huggingface

#set up language model using hugging face repository


repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.3,"max_new_tokens":2000})


    


# Define your prompt
template= """
You are an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's question
Question: {question}\n\nAnswer: Let's think step by step.
"""

prompt =PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

#create streamlit app
def main():
    st.title("Falcon llm Chatbot")
    
    #Get user input
    question = st.text_input("Enter your query")
    
    #Generate the response
    if st.button("Get answeer"):
        with st.spinner("Generating response..."):
            response = llm_chain.run(question)
        st.success(response)
if __name__=="__main__":
    main()










