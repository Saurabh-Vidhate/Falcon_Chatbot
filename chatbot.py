

import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer
from apikey import apikeys_huggingface
from langchain.schema import BaseOutputParser
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import re
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

#set huggingface API token

os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikeys_huggingface

#set up language model using hugging face repository


repo_id = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(repo_id)

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: list[list[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False

stop_tokens = [["Human", ":"], ["AI", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, device="cpu")]
)

import json

# Your existing StoppingCriteriaList
stopping_criteria = StoppingCriteriaList([StopGenerationCriteria(stop_tokens, tokenizer, device="cpu")])

# Extract relevant information for serialization
criteria_data = []
for criteria in stopping_criteria:
    criteria_data.append({
        "name": criteria.__class__.__name__,
        "data": {
            # Extract any relevant data here
        }
    })

# Serialize the criteria_data to JSON
json_serializable_data = json.dumps(criteria_data)

#load model
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.4, "stopping_criteria":json_serializable_data,
                                                    "repetition_penalty": 1.7,
                                                    "use_cache": True,
                                                    "bos_token_id": 1,
                                                    "eos_token_id": 11,
                                                    "pad_token_id": 11})


from sentence_transformers import SentenceTransformer, util
# Load a pre-trained sentence transformer model
sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Define predefined questions and answers (including synonyms)
predefined_responses = {
    "What's your company name?": "The company name is TCS.",
    "Tell me something about your company.": "TATA Consultancy Services Limited (TCS) is an Indian multinational information technology (IT) services and consulting company with its headquarters in Mumbai, Maharashtra.[6][7] It is a part of the Tata Group and operates in 150 locations across 46 countries",
    "Who is CEO of  your company ?": "K Krithivasan",
    "What is your company called?": "The company is called TCS.",
    "Please provide information about your organization.": "TATA Consultancy Services Limited (TCS) is an Indian multinational IT services company headquartered in Mumbai, Maharashtra.",
    "Who leads your organization?": "The CEO of our organization is K Krithivasan.",
    "what is company location ?":"It is located at Mumbai"
}


# Function to generate a custom response
def generate_custom_response(input_text):
    # Initialize a list to store similarity scores
    similarity_scores = []

    # Iterate through predefined questions and compute semantic similarity
    for question, answer in predefined_responses.items():
        similarity = util.pytorch_cos_sim(
            sentence_transformer_model.encode(input_text),
            sentence_transformer_model.encode(question)
        )[0][0]
        similarity_scores.append((question, similarity))

    # Sort similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # If a similar question is found with high similarity, return the corresponding answer
    if similarity_scores[0][1] > 0.7:
        return predefined_responses[similarity_scores[0][0]]
    else:
        # If no similar question is found, use the Falcon model
        text = input_text
        llm_response = chain(text)
        return llm_response["response"]

template = """
The following is a conversation between a human and an AI. The AI represents a Tata Consultancy Services (TCS)
company spokesperson and is knowledgeable about the company's details, products, and services.
It can provide information and answer questions related to the company. If Spokeperson does not know the
answer to a question, he truthfully says he does not know.



Current conversation:
{history}
Human: {input}
AI:
""".strip()

#define prompt
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

memory = ConversationBufferWindowMemory(
    memory_key="history", k=0, return_only_outputs=True
)

# Create your ConversationChain with LangChain and custom response generation
chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
)




#define output parser for clean response
class CleanupOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        user_pattern = r"\nUser"
        text = re.sub(user_pattern, "", text)
        human_pattern = r"\nHuman:"
        text = re.sub(human_pattern, "", text)
        ai_pattern = r"\nAI:"
        return re.sub(ai_pattern, "", text).strip()

    @property
    def _type(self) -> str:
        return "output_parser"

memory = ConversationBufferWindowMemory(
    memory_key="history", k=0, return_only_outputs=True
)

# Create your ConversationChain with LangChain and custom response generation
chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    output_parser=CleanupOutputParser(),
    verbose=True,
)



#create streamlit app
#create streamlit app
def main():
    st.title("Falcon llm Chatbot")

    #Get user input
    user_input = st.text_input("Enter your query")

    #Generate the response
    if st.button("Get answeer"):
        with st.spinner("Generating response..."):
            response = generate_custom_response(user_input)
        st.success(response)
if __name__=="__main__":
    main()
        