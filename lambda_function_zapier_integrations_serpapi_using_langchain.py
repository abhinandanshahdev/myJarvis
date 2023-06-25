import json
import os
import requests
import urllib.parse
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

def lambda_handler(event, context):
    
    print("Received event: ", event)
    
    openai_api_key = os.environ['OPENAI_API_KEY']

    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.9, model_name="gpt-4", max_tokens=500
    )
    
    serpapi_api_key = os.environ['SERP_API_KEY']
    
    zapier = ZapierNLAWrapper()
    
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)
   
    agent = initialize_agent(toolkit.get_tools()+tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # Get input from API Gateway and parse x-www-form-urlencoded data
    prompt = event['body']
    
    response = agent({prompt})
    
    print("Response from the agent: ", response)
    
    # Send the result to the webhook
    requests.post('https://twiliomessangerfunctionurl/myJarvisMessenger', json={'message': response['output']})

    # Return Twilio compatible response
    return {
        'statusCode': 200,
        'body': json.dumps({'message': "Your request is being processed. You will receive a response shortly."}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
}
