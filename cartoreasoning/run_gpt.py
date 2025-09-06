import os
import argparse

from dotenv import load_dotenv

import json
import pickle
import polars as pl

import base64           # To load image on local
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

with open('./instruction.pkl', 'rb') as handle:
    instructions = pickle.load(handle)
    instructions += "DO NOT use web search."    # Adding in case they use search
 



# INCOMPLETE
    
#     {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", 
    "body": {"model": "gpt-3.5-turbo-0125", 
             "messages": [{"role": "system", 
                           "content": "You are an unhelpful assistant."},
                           {"role": "user", "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
        ],}],"max_tokens": 512}}
    

    from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batchinput.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)

from openai import OpenAI
client = OpenAI()

batch_input_file_id = batch_input_file.id
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "nightly eval job"
    }
)

from openai import OpenAI
client = OpenAI()
# status checking
batch = client.batches.retrieve("batch_abc123")
print(batch)


client = OpenAI()

file_response = client.files.content("file-xyz123")
print(file_response.text)