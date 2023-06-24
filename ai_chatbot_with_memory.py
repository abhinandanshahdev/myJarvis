import json
import os
import boto3
import requests

from langchain.chat_models import ChatOpenAI
from langchain import ConversationChain
from langchain.prompts.prompt import PromptTemplate

s3 = boto3.resource('s3')

bucket = s3.Bucket('myjarvischathistorybucket') # replace with your actual bucket name

def lambda_handler(event, context):
    
    print("Received event: ", event)
    human_input = event['body']
    
    # Load chat history from S3
    chat_history_obj = bucket.Object('chat_history.json')
    try:
        chat_history = json.loads(chat_history_obj.get()['Body'].read().decode('utf-8'))
    except chat_history_obj.meta.client.exceptions.NoSuchKey:
        chat_history = ""

    template = f"""
    You are a helpful assistant.
    Your goal is to help the user whilst keeping your messages succinct
    
    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], 
        template=template
    )

    openai_api_key = os.environ['OPENAI_API_KEY']

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7, model_name="gpt-4", max_tokens=100
    )
    
    llm_chain = ConversationChain(
        llm=llm, 
        prompt=prompt, 
        verbose=True
    )

    response = llm_chain.predict(human_input=human_input)

    print("Response from the agent: ", response)
    
    # Update chat history and save it back to S3
    chat_history += "\nHuman: " + human_input + "\nChatbot: " + response
    chat_history_obj.put(Body=json.dumps(chat_history))

    # Send the result to the webhook
    requests.post('https://myjarvis-5485.twil.io/myJarvisMessenger', json={'message': response})

    # Return Twilio compatible response
    return {
        'statusCode': 200,
        'body': json.dumps({'message': "Your request is being processed. This is a placeholder response."}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
