import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

import tqdm 
import json  
import sklearn 
import scipy 

# load original messages (source of truth)
orig = pd.read_pickle("./Database/Messages.pkl")
orig = orig.copy()
orig['Message Id'] = orig.index

# make a unique mapping on Message (keep first if duplicates)
orig_unique = orig.drop_duplicates(subset='Message', keep='first')[['Message', 'Message Id']]

# load merged and merge in MessageId
merged_path = "./Database/Merged.pkl"
merged = pd.read_pickle(merged_path)

merged = merged.merge(orig_unique, on='Message', how='left')

# quick sanity checks
missing = merged['Message Id'].isna().sum()
print(f"Missing MessageId: {missing} / {len(merged)}")
if missing:
    print("Sample missing messages:", merged.loc[merged['Message Id'].isna(), 'Message'].head().tolist())

# save to a new file (or overwrite merged_path if you prefer)
merged.to_pickle("./Database/Merged_withMessageId.pkl")
print("Saved ./Database/Merged_withMessageId.pkl")

sources = {
    "BGE": "./Database/AdditionalEmbeddings/BGE/OpenEmailEmbeddings.pkl",
    "Gemma": "./Database/AdditionalEmbeddings/Gemma/OpenEmailEmbeddings.pkl",
    "Granite": "./Database/AdditionalEmbeddings/Granite/OpenEmailEmbeddings.pkl",
    "Qwen3-0.6B": "./Database/AdditionalEmbeddings/Qwen3-0.6B/OpenEmailEmbeddings.pkl",
    "Qwen3-4B": "./Database/AdditionalEmbeddings/Qwen3-4B/OpenEmailEmbeddings.pkl",
    "Qwen3-8B": "./Database/AdditionalEmbeddings/Qwen3-8B/OpenEmailEmbeddings.pkl",
    "OpenAI": "./Database/EmailEmbeddings.pkl",
}

dfs = []
for name, path in sources.items():
    df = pd.read_pickle(path)
    df["source"] = name            # optional: keep track of origin
    dfs.append(df)

email_embeddings = pd.concat(dfs, ignore_index=True)

sources = {
    "BGE": "./Database/AdditionalEmbeddings/BGE/OpenMessageEmbeddings.pkl",
    "Gemma": "./Database/AdditionalEmbeddings/Gemma/OpenMessageEmbeddings.pkl",
    "Granite": "./Database/AdditionalEmbeddings/Granite/OpenMessageEmbeddings.pkl",
    "Qwen3-0.6B": "./Database/AdditionalEmbeddings/Qwen3-0.6B/OpenMessageEmbeddings.pkl",
    "Qwen3-4B": "./Database/AdditionalEmbeddings/Qwen3-4B/OpenMessageEmbeddings.pkl",
    "Qwen3-8B": "./Database/AdditionalEmbeddings/Qwen3-8B/OpenMessageEmbeddings.pkl",
    "OpenAI": "./Database/MessageEmbeddings.pkl",
}

dfs = []
for name, path in sources.items():
    df = pd.read_pickle(path)
    df["source"] = name            # optional: keep track of origin
    dfs.append(df)

message_embeddings = pd.concat(dfs, ignore_index=True)

messages = pd.read_pickle("./Database/Merged_withMessageId.pkl")
"""
Index(['User Id', 'Experiment', 'Email Id', 'Phase Trial', 'Decision',
       'Message Num', 'Message', 'Email Type', 'Phase Value',
       'Experiment Condition', 'Categorization Confidence', 'Email Action',
       'Reaction Time', 'Correct Categorization', 'Role', 'Content',
       'Message Embedding', 'User Improvement', 'AI Generation Perception',
       'Pre Experiment Quiz Score', 'Response Message Similarity',
       'Email Embedding', 'Age', 'Gender', 'Education', 'Country', 'Victim',
       'Chatbot', 'Question 0', 'Question 1', 'Question 2', 'Question 3',
       'Question 4', 'Question 5', 'Post Experiment Question 1',
       'Post Experiment Question 2', 'Post Experiment Question 3',
       'Post Experiment Question 4', 'Open Response',
       'Open Response Embedding', 'Message Email Similarity',
       'User Initial Performance', 'User Final Performance', 'Gender Number',
       'Education Years', 'Phishing Experience', 'Chatbot Experience',
       'Cognitive Model Activity'],
      dtype='object')
"""
models = message_embeddings["Model"].unique()

current_models = ['./Models/bge-large-en-v1.5' ,'./Models/embeddinggemma-300m',
 './Models/granite-embedding-small-english-r2',
 './Models/qwen3-embedding-0.6B', './Models/Qwen3-Embedding-4B',
 './Models/Qwen3-Embedding-8B' ,'text-embedding-ada-002',
 'text-embedding-3-small' ,'text-embedding-3-large']

to_replace = ['BAAI BGE-Large', 'Google Gemma',
 'IBM Granite', 'Qwen3 0.6B' ,'Qwen3 4B',
 'Qwen3 8B', 'OpenAI ADA 2',
 'OpenAI Small 3', 'OpenAI Large 3']

# Build a mapping from old display names -> desired model keys and apply to both dataframes
mapping = dict(zip(current_models, to_replace))

# Replace model column values (no-op for values not in mapping)
message_embeddings['Model'] = message_embeddings['Model'].replace(mapping)
email_embeddings['Model'] = email_embeddings['Model'].replace(mapping)

# recompute models after replacement
message_embeddings = message_embeddings[message_embeddings["Model"] != 'Google Gemma']
models = message_embeddings["Model"].unique()

rows = []
total = len(models) * len(messages)

with tqdm.tqdm(total = total) as pbar:
    for model in models:
        model_message_embeddings = message_embeddings[message_embeddings["Model"] == model]
        model_email_embeddings = email_embeddings[email_embeddings["Model"] == model]

        similarity = []
        metrics = []
        for midx, message in messages.iterrows(): 
            message_embedding = model_message_embeddings[model_message_embeddings['MessageId'] == message['Message Id']]
            email_embedding = model_email_embeddings[model_email_embeddings['EmailId'] == message['Email Id']]
            
            if(len(message_embedding['Embedding']) == 0 or len(email_embedding['Embedding']) == 0): 
                pbar.update(1)
                continue 

            message_embedding = np.array(json.loads(message_embedding['Embedding'].item())).reshape(1, -1)
            email_embedding = np.array(json.loads(email_embedding['Embedding'].item())).reshape(1, -1)

            similarity.append(sklearn.metrics.pairwise.cosine_similarity(message_embedding, email_embedding)[0][0])
            metrics.append(message['Correct Categorization'])

            pbar.update(1)
        
        if(len(similarity) == 0 or len(metrics) == 0):
            continue

        # compute Pearson r and p-value between similarity and the human metric
        r, pval = scipy.stats.pearsonr(similarity, metrics)
        r2 = r**2
        Embedding_Size = len(json.loads(model_message_embeddings.iloc[0]['Embedding']))
        rows.append({
            "model": model,
            "r": np.abs(r),
            "r2": r2,
            "p": pval,
            "Embedding Size": Embedding_Size
        })

# create dataframe
size_accuracy = pd.DataFrame(rows)

# convert embedding size to log10 (safe-guard against zero)
size_accuracy['Log Embedding Size'] = size_accuracy['Embedding Size'].apply(lambda s: np.log10(s) if s and s > 0 else np.nan)

# drop rows with missing values for plotting/regression
plot_df = size_accuracy.dropna(subset=['r', 'Log Embedding Size']).copy()

plt.figure(figsize=(10,6))
# regression of per-model Pearson r (x) vs log embedding size (y)
ax = sns.regplot(data=plot_df, x="r", y="Log Embedding Size", ci=95, scatter=False, line_kws={"color":"C0"})
ax.scatter(plot_df["r"], plot_df["Log Embedding Size"], color="C1", s=50, zorder=5)

# annotate each point with model name, r^2 and p-value
for _, row in plot_df.iterrows():
    label = f"{row['model']}\nr\u00b2={row['r2']:.3f}, p={row['p']:.2g}"
    ax.annotate(label, (row['r'], row['Log Embedding Size']),
                textcoords="offset points", xytext=(6,6), fontsize=8, zorder=10)

# tighten axis limits to data extents with a small padding so regression line stays within view
if not plot_df.empty:
    x_min, x_max = plot_df['r'].min(), plot_df['r'].max()
    x_pad = max(0.01, 0.05 * (x_max - x_min if x_max > x_min else 1.0))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    y_min, y_max = plot_df['Log Embedding Size'].min(), plot_df['Log Embedding Size'].max()
    y_pad = max(0.05, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

# overall Pearson correlation between per-model r and log embedding size
if len(plot_df) >= 2:
    overall_r, overall_p = scipy.stats.pearsonr(plot_df["r"], plot_df["Log Embedding Size"])
    # place overall stats in the top-left inside the axes (axes coords)
    ax.text(0.02, 0.98, f"Overall r={overall_r:.3f}, p={overall_p:.2g}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8), fontsize=9)

ax.set_xlabel("Per-model Pearson r (similarity vs. human metric)")
ax.set_ylabel("Log10(Embedding Size)")
plt.tight_layout()
plt.show()
#

