"""
Azure OpenAI Message Embedding Pipeline

This script loads phishing experiment dataframes, cleans/parses chat messages, 
extracts message roles/contents, and computes message embeddings using the Azure OpenAI Embeddings API.  
The processed dataframe is saved with new 'Role', 'Content', and 'Embedding' columns as both a pickle and CSV file.

Key workflow:
- Load all relevant pickled dataframes.
- Clean and parse each message for JSON content; extract roles and messages.
- Define an OpenAI embedder using Azure API (with retry logic for rate limits).
- Compute an embedding for each cleaned message (tokenized/truncated as needed).
- Save the enriched Messages dataframe for downstream use.

Dependencies:
- pandas, tqdm, tiktoken, numpy, dotenv, openai[azure], azure-identity, python-dotenv
- Requires Azure OpenAI environment variables in .env (endpoint, key, deployment, etc.)
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import json
import time
import random

from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, RateLimitError, Timeout
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from tqdm import tqdm
import pandas as pd
import numpy as np
import tiktoken

# Load environment variables (Azure OpenAI keys/settings)
load_dotenv()

# Load experiment dataframes (all as pickles)
Annotations = pd.read_pickle("./Database/Annotations.pkl")
Demographics = pd.read_pickle("./Database/Demographics.pkl")
Emails = pd.read_pickle("./Database/Emails.pkl")
Embeddings = pd.read_pickle("./Database/Embeddings.pkl")
Messages = pd.read_pickle("./Database/Messages.pkl")

def clean_message(text):
    """
    Cleans up message text for JSON parsing:
    - Replaces curly/smart quotes with straight quotes.
    - Escapes invalid single backslashes.
    - Removes non-ASCII characters.
    """
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r'\\(?![\\\'"abfnrtv0-9xuU])', r'\\\\', text)
    text = text.encode('ascii', errors='ignore').decode()
    return text

# Extract message roles and contents (parsed from JSON if possible, else fallback to raw string)
roles = []
contents = []
for _, message in Messages.iterrows():
    cleaned = clean_message(message['Message'])
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {'role': 'system', 'content': cleaned}
    roles.append(parsed.get('role', None))
    contents.append(parsed.get('content', None))

Messages['Role'] = roles
Messages['Content'] = contents

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

if __name__ == "__main__":
    # Get Azure OpenAI settings from environment variables
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")  
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    MODEL_NAME = os.getenv("MODEL_NAME")
    MAX_TOKENS = 8192

    # Instantiate embedder
    embedder = OpenAIEmbedder(
        azure_endpoint=AZURE_ENDPOINT, 
        azure_api_key=AZURE_API_KEY, 
        azure_development_name=AZURE_DEPLOYMENT_NAME, 
        azure_api_version=AZURE_API_VERSION, 
        model_name=MODEL_NAME, 
        max_tokens=MAX_TOKENS
    )

    # Get all cleaned message contents as a list (fill NAs as empty string)
    texts = Messages['Content'].fillna('').tolist()

    # Compute embeddings for each message (single threaded for API safety)
    embeddings = []
    for i in tqdm(range(0, len(texts), 1), desc="Embedding"):
        batch = texts[i]
        embeddings.append(embedder.embed(batch))

    # Add the new embedding column and save to disk
    Messages['Embedding'] = embeddings

    Messages.to_pickle("./Database/MessageEmbeddings.pkl")
    Messages.to_csv("./Database/MessageEmbeddings.csv")