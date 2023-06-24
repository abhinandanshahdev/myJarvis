import json
import os
import requests
import openai
import urllib.parse

def lambda_handler(event, context):
    print("Received event: ", event)

    # Set the OpenAI API key
    openai.api_key = os.environ['OPENAI_API_KEY']

    # Get input from API Gateway and parse x-www-form-urlencoded data
    prompt = event['body']

    # Construct the messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]

    print("Sending these messages to OpenAI API: ", messages)

    # Make API request
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    # Extract the generated text
    text = response['choices'][0]['message']['content']
    
    # Send the result to the webhook
    requests.post('https://myjarvis-5485.twil.io/myJarvisMessenger', json={'message': text})

    # Return Twilio compatible response
    return {
        'statusCode': 200,
        'body': json.dumps({'message': "Your request is being processed. This is a placeholder response."}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
