"""
This comparison splits the users into 'high' and 'low' learning Improvement groups and compares the embeddings of the messages sent between those users and the ChatBot to see if there is any difference. 
"""
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity

MessageEmbeddings = pd.read_pickle('./Database/MessageEmbeddings.pkl')
"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct', 'Role', 'Content', 'Embedding'],
      dtype='object')
"""
MessageEmbeddings = MessageEmbeddings[MessageEmbeddings['Role'] == 'system']
Annotations = pd.read_pickle("./Database/Annotations.pkl")
"""
Index(['UserId', 'Experiment', 'ExperimentCondition', 'EmailId', 'PhaseTrial',
       'Decision', 'EmailType', 'PhaseValue', 'Confidence', 'EmailAction',
       'ReactionTime', 'Correct'],
      dtype='object')
"""
Embeddings = pd.read_pickle("./Database/Embeddings.pkl")
"""
Index(['EmailId', 'BaseEmailID', 'Author', 'Style', 'Embedding',
       'Phishing Similarity', 'Ham Similarity'],
      dtype='object')
"""

High_Improvements = []
Low_Improvements = []
High_Performances = []
Low_Performances = []
User_Improvements = []
User_Performances = []
Email_Similarities = []

for idx, messageEmbedding in MessageEmbeddings.iterrows():
    UserAnnotations = Annotations[Annotations['UserId'] == messageEmbedding['UserId']]
    preTrainingAccuracy = UserAnnotations[UserAnnotations['PhaseValue'] == "preTraining"]['Correct'].mean()
    postTrainingAccuracy = UserAnnotations[UserAnnotations['PhaseValue'] == "postTraining"]['Correct'].mean()
    User_Improvement = int(round(((100 * (postTrainingAccuracy - preTrainingAccuracy)) / 10) * 10))
    Email_Embedding = Embeddings[Embeddings['EmailId'] == messageEmbedding['EmailId']]['Embedding']
    email_vec = np.array(Email_Embedding.values[0]).reshape(1, -1)
    msg_vec = np.array(messageEmbedding['Embedding']).reshape(1, -1)
    
    Email_Similarity = cosine_similarity(email_vec, msg_vec)[0][0]

    if(User_Improvement > 0.2):
        High_Improvements.append(1)
    else:
        High_Improvements.append(0)
    
    if(User_Improvement < 0.2):
        Low_Improvements.append(1)
    else: 
        Low_Improvements.append(0)

    User_Improvements.append(User_Improvement)

    User_Performance = UserAnnotations['Correct'].mean()

    if(User_Performance > 0.8):
        High_Performances.append(1)
    else:
        High_Performances.append(0)
    
    if(User_Improvement < 0.8):
        Low_Performances.append(1)
    else: 
        Low_Performances.append(0)

    User_Performances.append(User_Performance)
    Email_Similarities.append(Email_Similarity)

MessageEmbeddings['Low_Improvement'] = Low_Improvements
MessageEmbeddings["High_Improvement"] = High_Improvements
MessageEmbeddings["User_Improvement"] = User_Improvements

MessageEmbeddings['Low_Performances'] = Low_Performances
MessageEmbeddings["High_Performance"] = High_Performances
MessageEmbeddings["User_Performance"] = User_Performances
MessageEmbeddings["Email_Similarity"] = Email_Similarities

MessageEmbeddings.to_pickle("./Database/MessageEmbeddings.pkl")
MessageEmbeddings.to_csv("./Database/MessageEmbeddings.csv")

# Select only the rows where 'High Improvement' is True
high_embeds = MessageEmbeddings.loc[MessageEmbeddings["High_Improvement"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
high_embeds_stack = np.stack(high_embeds.values)
# Compute the mean embedding
mean_high_Improvement_embeddings = high_embeds_stack.mean(axis=0).reshape(1, -1)

# Select only the rows where 'Low_Improvements' is True
low_embeds = MessageEmbeddings.loc[MessageEmbeddings["Low_Improvement"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
low_embeds_stack = np.stack(low_embeds.values)
# Compute the mean embedding
mean_low_Improvement_embeddings = low_embeds_stack.mean(axis=0).reshape(1, -1)

# Select only the rows where 'High Performance' is True
high_embeds = MessageEmbeddings.loc[MessageEmbeddings["High_Performance"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
high_embeds_stack = np.stack(high_embeds.values)
# Compute the mean embedding
mean_high_performance_embeddings = high_embeds_stack.mean(axis=0).reshape(1, -1)

# Select only the rows where 'Low_Performances' is True
low_embeds = MessageEmbeddings.loc[MessageEmbeddings["Low_Performances"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
low_embeds_stack = np.stack(low_embeds.values)
# Compute the mean embedding
mean_low_performance_embeddings = low_embeds_stack.mean(axis=0).reshape(1, -1)

columns = ["User_Improvement",
            "High_Improvement_Embedding_Similarity",
            "Low_Improvements Embedding Similarity",
            "User_Performance",
            "High_Performance_Embedding_Similarity",
            "Low_Performance_Embedding_Similarity",
            "Email_Similarity"]

df = pd.DataFrame([], columns=columns)

for UserId in MessageEmbeddings['UserId'].unique():
    UserMessages = MessageEmbeddings[MessageEmbeddings['UserId'] == UserId]

    User_Embeddings = UserMessages["Embedding"]
    User_Embeddings_Stack = np.stack(User_Embeddings.values)
    User_Mean_Embedding = User_Embeddings_Stack.mean(axis=0).reshape(1, -1)

    High_Improvement_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_high_Improvement_embeddings)[0][0]
    Low_Improvement_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_low_Improvement_embeddings)[0][0]
    High_Performance_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_high_performance_embeddings)[0][0]
    Low_Performance_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_low_performance_embeddings)[0][0]
    User_Improvement = UserMessages['User_Improvement'].mean()
    User_Performance = UserMessages['User_Performance'].mean() * 100
    Email_Similarity = UserMessages['Email_Similarity'].mean()

    d = pd.DataFrame([{ "User_Improvement":User_Improvement,
                        "High_Improvement_Embedding_Similarity":High_Improvement_Embedding_Similarity,
                        "Low_Improvements Embedding Similarity":Low_Improvement_Embedding_Similarity,
                        "User_Performance":User_Performance,
                        "High_Performance_Embedding_Similarity":High_Performance_Embedding_Similarity,
                        "Low_Performance_Embedding_Similarity":Low_Performance_Embedding_Similarity,
                        "Email_Similarity":Email_Similarity
                    }])
    
    if(len(df) == 0): 
        df = d
    else:
        df = pd.concat([df,d])

df = df[df["High_Performance_Embedding_Similarity"] > 0.925] 
# Convert all needed columns to numeric and drop any invalid rows
cols = [
    "Email_Similarity",
    "User_Performance",
    "User_Improvement",  # aka "User_Improvement"
    "High_Improvement_Embedding_Similarity",
    "High_Performance_Embedding_Similarity"
]

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# Set up a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. High_Improvement_Embedding_Similarity vs User_Improvement
x1 = df["High_Improvement_Embedding_Similarity"]
y1 = df["User_Improvement"]
sns.regplot(x=x1, y=y1, ax=axes[0][0])
slope, intercept, r_val, p_val, _ = linregress(x1, y1)
axes[0][0].text(0.05, 0.95, f'$R^2$ = {r_val**2:.2f}\np = {p_val:.4f}', transform=axes[0][0].transAxes, fontsize=12, verticalalignment='top')
axes[0][0].set_title("Feedback Embedding Similarity\nto High Improvement User's Feedback Embeddings \nvs User Pre-Post Training Improvement", fontsize=16)
axes[0][0].set_xlabel("")
axes[0][0].set_ylabel("User_Improvement (Percentage Points)", fontsize=14)

# 2. Email_Similarity vs User_Improvement
x2 = df["Email_Similarity"]
y2 = df["User_Improvement"]
sns.regplot(x=x2, y=y2, ax=axes[0][1])
slope, intercept, r_val, p_val, _ = linregress(x2, y2)
axes[0][1].text(0.05, 0.95, f'$R^2$ = {r_val**2:.2f}\np = {p_val:.4f}', transform=axes[0][1].transAxes, fontsize=12, verticalalignment='top')
axes[0][1].set_title("Email and Feedback Embedding Similarity \nvs User Pre-Post Training Improvement", fontsize=16)
axes[0][1].set_xlabel("")
axes[0][1].set_ylabel("")

# 3. High_Performance_Embedding_Similarity vs User_Performance
x3 = df["High_Performance_Embedding_Similarity"]
y3 = df["User_Performance"]
sns.regplot(x=x3, y=y3, ax=axes[1][0])
slope, intercept, r_val, p_val, _ = linregress(x3, y3)
axes[1][0].text(0.05, 0.95, f'$R^2$ = {r_val**2:.2f}\np = {p_val:.4f}', transform=axes[1][0].transAxes, fontsize=12, verticalalignment='top')
axes[1][0].set_title("Feedback Embedding Similarity\nto High Performance User's Feedback Embeddings \nvs User Total Mean Performance", fontsize=16)
axes[1][0].set_xlabel("Embedding Similarity", fontsize=14)
axes[1][0].set_ylabel("User_Performance (Percentage)", fontsize=14)

# 4. Email_Similarity vs User_Performance
x4 = df["Email_Similarity"]
y4 = df["User_Performance"]
sns.regplot(x=x4, y=y4, ax=axes[1][1])
slope, intercept, r_val, p_val, _ = linregress(x4, y4)
axes[1][1].text(0.05, 0.95, f'$R^2$ = {r_val**2:.2f}\np = {p_val:.4f}', transform=axes[1][1].transAxes, fontsize=12, verticalalignment='top')
axes[1][1].set_title("Email and Feedback Embedding Similarity \nvs User Total Mean Performance", fontsize=16)
axes[1][1].set_xlabel("Embedding Similarity", fontsize=14)
axes[1][1].set_ylabel("")

plt.tight_layout()
plt.show()