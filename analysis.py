import ast 
import re 
import json 
from tqdm import tqdm
import pandas as pd 
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

Annotations = pd.read_pickle("./Database/Annotations.pkl")
"""
Index(['UserId', 'Experiment', 'ExperimentCondition', 'EmailId', 'PhaseTrial',
       'Decision', 'EmailType', 'PhaseValue', 'Confidence', 'EmailAction',
       'ReactionTime', 'Correct'],
      dtype='object')
"""
Demographics = pd.read_pickle("./Database/Demographics.pkl")
"""
Index(['UserId', 'Age', 'Gender', 'Education', 'Country', 'Victim', 'Chatbot',
       'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5'],
      dtype='object')
"""

Emails = pd.read_pickle("./Database/Emails.pkl")
"""
Index(['EmailId', 'BaseEmailID', 'Author', 'Style', 'Type', 'Sender Style',
       'Sender', 'Subject', 'Sender Mismatch', 'Request Credentials',
       'Subject Suspicious', 'Urgent', 'Offer', 'Link Mismatch', 'Prompt',
       'Body'],
      dtype='object')
"""

Embeddings = pd.read_pickle("./Database/Embeddings.pkl")
"""
Index(['EmailId', 'BaseEmailID', 'Author', 'Style', 'Embedding',
       'Phishing Similarity', 'Ham Similarity'],
      dtype='object')
"""

Messages = pd.read_pickle("./Database/Messages.pkl")
"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct'],
      dtype='object')
"""

def clean_message(text):
    # Replace curly quotes with straight quotes
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    
    # Escape only invalid backslashes
    text = re.sub(r'\\(?![\\\'"abfnrtv0-9xuU])', r'\\\\', text)
    
    # Remove non-ASCII characters
    text = text.encode('ascii', errors='ignore').decode()

    return text

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

# Add new columns to Messages
Messages['Role'] = roles
Messages['Content'] = contents

#print(Messages.columns)
"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct', 'Role', 'Content'],
      dtype='object')
"""

import torch
from transformers import AutoTokenizer, AutoModel

class HFEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed(self, texts, batch_size=8):
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

# pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
embedder = HFEmbedder()  

texts = Messages['Content'].fillna('').tolist()

embeddings = []
for i in tqdm(range(0, len(texts), 8), desc="Embedding"):
    batch = texts[i:i + 8]
    embeddings.extend(embedder.embed(batch))

# Add to dataframe
Messages['Embedding'] = embeddings

Messages.to_pickle("./Database/MessageEmbeddings.pkl")
Messages.to_csv("./Database/MessageEmbeddings.csv")