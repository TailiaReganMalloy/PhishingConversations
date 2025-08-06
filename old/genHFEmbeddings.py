"""
Message Embedding Pipeline

This script loads multiple related datasets for a phishing email experiment, cleans and parses chat messages, 
extracts message roles and contents, and computes sentence embeddings for each message using a HuggingFace transformer model. 
The enriched dataframe (with roles, contents, and embeddings) is saved to disk as both a pickle and CSV file.

Key steps:
- Loads dataframes (annotations, demographics, emails, existing embeddings, and messages) from pickles.
- Cleans messages for JSON parsing (removing non-ASCII, fixing quotes, etc.).
- Extracts message role/content from JSON (or uses fallback).
- Defines a simple HuggingFace-based embedder using mean pooling.
- Batches and computes message embeddings.
- Saves the enriched Messages dataframe with embeddings.

Dependencies:
- pandas, torch, tqdm, transformers, sentence_transformers, dotenv
"""

import ast 
import re 
import json 
from tqdm import tqdm
import pandas as pd 
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables, if any are needed
load_dotenv()

# Load all relevant datasets
Annotations = pd.read_pickle("./Database/Annotations.pkl")
Demographics = pd.read_pickle("./Database/Demographics.pkl")
Emails = pd.read_pickle("./Database/Emails.pkl")
Embeddings = pd.read_pickle("./Database/Embeddings.pkl")
Messages = pd.read_pickle("./Database/Messages.pkl")

def clean_message(text):
    """
    Cleans up message text for JSON parsing:
    - Replaces curly quotes with straight quotes.
    - Escapes invalid backslashes.
    - Removes non-ASCII characters.
    """
    # Replace curly quotes with straight quotes
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    # Escape only invalid backslashes
    text = re.sub(r'\\(?![\\\'"abfnrtv0-9xuU])', r'\\\\', text)
    # Remove non-ASCII characters
    text = text.encode('ascii', errors='ignore').decode()
    return text

# Parse each message, extracting role/content from JSON if possible, else fallback to raw text
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

# Add extracted role/content columns
Messages['Role'] = roles
Messages['Content'] = contents

import torch
from transformers import AutoTokenizer, AutoModel

class HFEmbedder:
    """
    Simple HuggingFace embedding wrapper for sentence embedding extraction.
    Uses mean pooling over token embeddings.
    """
    def __init__(self, model_name="Qwen/Qwen3-0.6B", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed(self, texts, batch_size=8):
        """
        Batches and embeds a list of texts, returning mean-pooled embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded.attention_mask.unsqueeze(-1)

                # Mean pooling
                summed = (token_embeddings * attention_mask).sum(dim=1)
                count = attention_mask.sum(dim=1)
                mean_pooled = summed / count

                embeddings.extend(mean_pooled.cpu().tolist())
        return embeddings

# Instantiate embedder (edit model as needed)
embedder = HFEmbedder()  

texts = Messages['Content'].fillna('').tolist()

# Batch embed all message contents with progress bar
embeddings = []
for i in tqdm(range(0, len(texts), 8), desc="Embedding"):
    batch = texts[i:i + 8]
    embeddings.extend(embedder.embed(batch))

# Add embeddings to the dataframe
Messages['Embedding'] = embeddings

# Save enriched dataframe to pickle and CSV
Messages.to_pickle("./Database/MessageEmbeddings.pkl")
Messages.to_csv("./Database/MessageEmbeddings.csv")