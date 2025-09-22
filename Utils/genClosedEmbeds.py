import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import json
import time
import random

from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, RateLimitError, Timeout

import pandas as pd
import torch as th
import numpy as np
import json

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 
import tiktoken

load_dotenv()

Emails = pd.read_pickle("../Database/Emails.pkl")
Messages = pd.read_pickle("../Database/Messages.pkl")
Messages['MessageId'] = Messages.index 


emailColumns = ["EmailId", "Model", "Embedding"]
OpenEmailEmbeddings = pd.DataFrame([], columns=emailColumns)
messageColumns = ["MessageId", "Model", "Embedding"]
OpenMessageEmbeddings = pd.DataFrame([], columns=messageColumns)

class OpenAIEmbedder:
    """
    Embeds texts using Azure OpenAI Embeddings API, with automatic retry logic and tokenization/truncation.
    """
    def __init__(self, azure_endpoint, azure_api_key, azure_development_name, azure_api_version, model_name, max_tokens):
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_development_name = azure_development_name
        self.azure_api_version = azure_api_version
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
        )

    def safe_embedding(self, *args, **kwargs):
        """
        Calls OpenAI embeddings endpoint with retries for rate limit or API errors.
        """
        max_retries = 5
        for retry in range(max_retries):
            try:
                return self.client.embeddings.create(*args, **kwargs)
            except (RateLimitError, APIError, Timeout) as e:
                wait = (2 ** retry) + random.uniform(0, 1)
                print(f"[Worker] Rate limit or API error (embedding): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"[Worker] Unexpected error (embedding): {e}")
                raise
        raise RuntimeError(f"Failed after {max_retries} retries.")
    
    def embed(self, texts, *args, **kwargs):
        """
        Embeds a single string by:
        - Tokenizing and truncating to max_tokens if needed.
        - Sending to Azure OpenAI Embeddings endpoint.
        Returns: embedding vector as list.
        """
        encoding = tiktoken.encoding_for_model(self.model_name)
        tokens = encoding.encode(texts)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        truncated_string = encoding.decode(tokens)
        return self.safe_embedding(input=[truncated_string], model=self.model_name).data[0].embedding

# Get Azure OpenAI settings from environment variables
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")  
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_TOKENS = 8192

# 'text-embedding-3-large', 'text-embedding-3-small',
model_names = [ 'text-embedding-ada-002']

emailColumns = ["EmailId", "Model", "Embedding"]
OpenEmailEmbeddings = pd.DataFrame([], columns=emailColumns)
messageColumns = ["MessageId", "Model", "Embedding"]
OpenMessageEmbeddings = pd.DataFrame([], columns=messageColumns)

total = len(model_names) * (len(Messages) + len(Emails))

with tqdm(total = total) as pbar:
    for model_name in model_names:
        embedder = OpenAIEmbedder(
            azure_endpoint=AZURE_ENDPOINT, 
            azure_api_key=AZURE_API_KEY, 
            azure_development_name=AZURE_DEPLOYMENT_NAME, 
            azure_api_version=AZURE_API_VERSION, 
            model_name=model_name, 
            max_tokens=MAX_TOKENS
        )
        
        counter = 0
        for idx, col in Emails.iterrows():
            counter += 1
            #if counter > 10: continue 
            Text = ("Sender: " + col['Sender'] + "\n" + 
                        "Subject" + col['Subject'] + "\n" + 
                        col["Body"])
            embedding = embedder.embed(Text)
            embedding_json = json.dumps(embedding)
            d = pd.DataFrame([[col['EmailId'], model_name, embedding_json]], columns=emailColumns)
            if(len(OpenEmailEmbeddings) == 0):
                OpenEmailEmbeddings = d 
            else:
                OpenEmailEmbeddings = pd.concat([d, OpenEmailEmbeddings], ignore_index=True)

            if(counter % 1000):
                OpenMessageEmbeddings.to_pickle("../Database/Snapshots/MessageEmbeddings.pkl")
                OpenEmailEmbeddings.to_pickle("../Database/Snapshots/EmailEmbeddings.pkl")
                OpenMessageEmbeddings.to_csv("../Database/Snapshots/MessageEmbeddings.csv")
                OpenEmailEmbeddings.to_csv("../Database/Snapshots/EmailEmbeddings.csv")
            
            pbar.update(1)

        counter = 0
        for idx, col in Messages.iterrows(): 
            counter += 1
            #if counter > 10: continue 
            embedding = embedder.embed(col['Message'])
            embedding_json = json.dumps(embedding)

            d = pd.DataFrame([[col['MessageId'], model_name, embedding_json]], columns=messageColumns)
            if(len(OpenMessageEmbeddings) == 0):
                OpenMessageEmbeddings = d 
            else:
                OpenMessageEmbeddings = pd.concat([d, OpenMessageEmbeddings], ignore_index=True)
            
            pbar.update(1)

            if(counter % 1000):
                OpenMessageEmbeddings.to_pickle("../Database/Snapshots/MessageEmbeddings.pkl")
                OpenEmailEmbeddings.to_pickle("../Database/Snapshots/EmailEmbeddings.pkl")
                OpenMessageEmbeddings.to_csv("../Database/Snapshots/MessageEmbeddings.csv")
                OpenEmailEmbeddings.to_csv("../Database/Snapshots/EmailEmbeddings.csv")


OpenMessageEmbeddings.to_pickle("../Database/MessageEmbeddings.pkl")
OpenEmailEmbeddings.to_pickle("../Database/EmailEmbeddings.pkl")
OpenMessageEmbeddings.to_csv("../Database/MessageEmbeddings.csv")
OpenEmailEmbeddings.to_csv("../Database/EmailEmbeddings.csv")
