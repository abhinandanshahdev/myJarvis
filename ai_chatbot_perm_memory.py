import json
import os
import requests
import urllib.parse
import langchain
import pinecone
import openai
#import langchain related tools
from langchain.agents import load_tools
from langchain.tools import Tool
from langchain.tools.zapier.tool import ZapierNLARunAction
from langchain.agents import initialize_agent, ZeroShotAgent, AgentExecutor
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain
#import pinecone related stuff
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from datetime import datetime


def lambda_handler(event, context):
    
    
    print("Received event: ", event)
    
    
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0, model_name="gpt-3.5-turbo-0613", max_tokens=400
    )
    
    search = GoogleSearchAPIWrapper()
    google_api_key = os.environ['GOOGLE_API_KEY']
    cse_api_key = os.environ['GOOGLE_CSE_ID']
    
    #retreive pinecone api key and environment from env variables 
    pinecone_api_key = os.environ['PINECONE_API_KEY']
    pinecone_env = os.environ['PINECONE_ENV']
    openai.api_key = os.environ['OPENAI_API_KEY']
    
    # initialise pinecone
    pinecone.init(api_key=pinecone_api_key,
    environment=pinecone_env
    )
    
    # pinecone related functions
    def create_pinecone_index(table_name, dimension=1536, metric="cosine", pod_type="p1"):
        if table_name not in pinecone.list_indexes():
            print("Pinecone index does not exist")
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

    def get_ada_embedding(text):
        print("creating embedding using openai embedding model")
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]

    def upsert_to_index(index, texts):
        print("inserting into vector database")
        pinecone_vectors = []
        for loopIndex, text in enumerate(texts, start=1):
            pinecone_vectors.append(("test-openai-"+str(loopIndex), get_ada_embedding(text), {"text": text}))
        index.upsert(vectors=pinecone_vectors)
    
    def query_index(index, query_text, top_k=3):
        print("Querying pinecone index for top 3 results")
        q_embedding = get_ada_embedding(query_text)
        pineQ = index.query(q_embedding, top_k=top_k, include_values=False, include_metadata=True)
        return pineQ
    
    def print_results(pineQ):
        print("printing pinecone index search results")
        print(f"\033[36m" + str(pineQ) + "\033[0m")
        print("\n")
        for match in pineQ.matches:
            print(f"\033[1m\033[32m" + match.metadata['text'] + " (" + str(round(match.score*100,2)) + "%)" + "\033[0m")
    
    #different table for different users
    pinecone_table = "8007999709"
    
    #create the table if it doesnt exist 
    print("creating pinecone index, if it doesnt exit.")
    create_pinecone_index(pinecone_table)
    
    #get a reference to the created table
    index = pinecone.Index(pinecone_table)

    embedding_fn = OpenAIEmbeddings().embed_query

    vectorstore = Pinecone(index, embedding_fn, "text")
    
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
    
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    
    # the retreiver will retrieve the relevant history associated with the user input and prints it
    history = memory.load_memory_variables({"prompt": event['body']})["history"]
    
    print("The Retriever has returned following relevant history: " + history)

    input = event['body']
    
    agent_scratchpad = " "
    
    #initialise Zapier NLA wrapper 
    
    zapier = ZapierNLAWrapper() 
    
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    
    search = GoogleSearchAPIWrapper()

    tool = Tool(
        name="GoogleSearch",
        description="Search Google for things that require access to the internet for instance latest events, or news, otherwise do not use this tool",
        func=search.run,
    )

    def send_message_to_webhook(message, sender):
        url = 'your twilio serverless function'
        requests.post(url, json={'message': message, 'sender': sender})
        
    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and Jarvis, an AI. Jarvis is talkative and is a behavioural psychologist, he provides lots of specific details from its context. If Jarvis does not know the answer to a question, he truthfully says he does not know. He addresses the human by name when he can, 

    Relevant pieces of previous conversation:
    {history}
    
    (You do not need to use these pieces of information if not relevant)
    
    Current conversation:
    Human: {input}
    AI:"""
    prompt_new = PromptTemplate(
        input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
    )
    
    
    #agent = initialize_agent(toolkit.get_tools()+[tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    #new code for using vectorstore memory
    
    conversation_with_summary = ConversationChain(
    llm=llm, 
    prompt=prompt_new,
    # We set a very low max_token_limit for the purposes of testing.
    memory=memory,
    verbose=True
    )

    # Get input from API Gateway and parse x-www-form-urlencoded data
   
    sender = event['phoneNumber']
    
    
    #response = agent.run({prompt}) 
    
    response = conversation_with_summary.predict(input=input)

    
    print("Response from the agent: ", response)
    # Send the result to the webhook
    
    
    # Split the response into chunks of 1599 characters
    chunks = [response[i:i + 1599] for i in range(0, len(response), 1599)]

    # Send each chunk as a separate message
    for chunk in chunks:
        send_message_to_webhook(chunk, sender)
        
    # Return Twilio compatible response
    return {
        'statusCode': 200,
        'body': json.dumps({'message': "Your request is being processed. You will receive a response shortly."}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
}
