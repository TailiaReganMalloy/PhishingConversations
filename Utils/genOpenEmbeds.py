import pandas as pd
import torch as th
import numpy as np
import json

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 

Emails = pd.read_pickle("../Database/OriginalPaper/Emails.pkl")
Messages = pd.read_pickle("../Database/Messages.pkl")
Messages['MessageId'] = Messages.index 

# These are the commands to download local versions of these embedding models: 
# huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./Models/qwen3-embedding-0.6B
# Qwen 
# huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir ./Models/bge-large-en-v1.5
# huggingface-cli download ibm-granite/granite-embedding-small-english-r2 --local-dir ./Models/granite-embedding-small-english-r2
# huggingface-cli download google/embeddinggemma-300m --local-dir ./Models/embeddinggemma-300m
# huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir ./Models/Qwen3-Embedding-8B
# huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir ./Models/Qwen3-Embedding-4B
# huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./Models/MiniLM-L6-v2 ssh tailia@trux-hayabusa.uni.lux wsDP0RJPLhd2IwZ


# rsync -avP ./Database/ tailia@trux-hayabusa.uni.lux:/home/tailia/PhishingConversations/Database/ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

emailColumns = ["EmailId", "Model", "Embedding"]
OpenEmailEmbeddings = pd.DataFrame([], columns=emailColumns)
messageColumns = ["MessageId", "Model", "Embedding"]
OpenMessageEmbeddings = pd.DataFrame([], columns=messageColumns)

#model_paths = ["./Models/qwen3-embedding-0.6B", "./Models/Qwen3-Embedding-4B", "./Models/Qwen3-Embedding-8B", "./Models/bge-large-en-v1.5", "./Models/granite-embedding-small-english-r2", "./Models/embeddinggemma-300m" ]
model_paths = ["./Models/Qwen3-Embedding-4B"]
total = (len(Messages) + len(Emails)) * len(model_paths)

def extract_embedding(model, outputs, inputs, normalize=True):
    # outputs: model(**inputs, return_dict=True)
    # inputs: tokenizer(..., return_tensors="pt")
    # returns: 1D numpy float32 embedding
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output.squeeze(0)
    elif hasattr(outputs, "sentence_embedding") and outputs.sentence_embedding is not None:
        emb = outputs.sentence_embedding.squeeze(0)
    elif isinstance(outputs, dict) and outputs.get("last_hidden_state") is not None and outputs.get("last_hidden_state").ndim == 3:
        token_embeds = outputs.last_hidden_state  # (1, seq_len, dim)
        attn = inputs.get("attention_mask")
        if attn is None:
            emb = token_embeds.mean(dim=1).squeeze(0)
        else:
            mask = attn.unsqueeze(-1).type_as(token_embeds)
            summed = (token_embeds * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = (summed / counts).squeeze(0)
    else:
        raise RuntimeError("No usable embedding found in model outputs")
    emb = emb.cpu().detach().numpy().astype(np.float32)
    if normalize:
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
    return emb

# force CPU-only
device = "cpu"

with tqdm(total = total) as pbar:
    for model_path in model_paths:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # CPU-only safe load
        model = AutoModel.from_pretrained(
            model_path,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.to("cpu")
        model.eval()

        with th.no_grad():
            # Do this again for open responses? 
            for idx, col in Emails.iterrows():
                Text = ("Sender: " + col['Sender'] + "\n" + 
                        "Subject" + col['Subject'] + "\n" + 
                        col["Body"])
                
                # truncate to model/tokenizer max length and send tensors to model device
                max_len = getattr(tokenizer, "model_max_length", None)
                if max_len is None or max_len <= 0:
                    max_len = getattr(getattr(model, "config", None), "max_position_embeddings", 512)
                inputs = tokenizer(Text, return_tensors="pt", truncation=True, padding=True, max_length=int(max_len))
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, return_dict=True)
                embedding = extract_embedding(model, outputs, inputs)
                # JSON-serialize embedding for CSV-safe, non-truncated representation
                embedding_json = json.dumps(embedding.tolist())

                d = pd.DataFrame([[col['EmailId'], model_path, embedding_json]], columns=emailColumns)
                if(len(OpenEmailEmbeddings) == 0):
                    OpenEmailEmbeddings = d 
                else:
                    OpenEmailEmbeddings = pd.concat([d, OpenEmailEmbeddings], ignore_index=True)
                
                pbar.update(1)
            for idx, col in Messages.iterrows():
                Text = col['Message']
                # truncate to model/tokenizer max length and send tensors to model device
                max_len = getattr(tokenizer, "model_max_length", None)
                if max_len is None or max_len <= 0:
                    max_len = getattr(getattr(model, "config", None), "max_position_embeddings", 512)
                inputs = tokenizer(Text, return_tensors="pt", truncation=True, padding=True, max_length=int(max_len))
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, return_dict=True)
                embedding = extract_embedding(model, outputs, inputs)
                # JSON-serialize embedding for CSV-safe, non-truncated representation
                embedding_json = json.dumps(embedding.tolist())

                d = pd.DataFrame([[col['MessageId'], model_path, embedding_json]], columns=messageColumns)
                if(len(OpenMessageEmbeddings) == 0):
                    OpenMessageEmbeddings = d 
                else:
                    OpenMessageEmbeddings = pd.concat([d, OpenMessageEmbeddings], ignore_index=True)
                
                pbar.update(1)

OpenMessageEmbeddings.to_pickle("../Database/OpenMessageEmbeddings.pkl")
OpenEmailEmbeddings.to_pickle("../Database/OpenEmailEmbeddings.pkl")
OpenMessageEmbeddings.to_csv("../Database/OpenMessageEmbeddings.csv")
OpenEmailEmbeddings.to_csv("../Database/OpenEmailEmbeddings.csv")
