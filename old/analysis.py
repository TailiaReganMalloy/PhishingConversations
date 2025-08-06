"""
This comparison splits the users into 'high' and 'low' learning outcome groups and compares the embeddings of the messages sent between those users and the ChatBot to see if there is any difference. 
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
Annotations = pd.read_pickle("./Database/Annotations.pkl")
"""
Index(['UserId', 'Experiment', 'ExperimentCondition', 'EmailId', 'PhaseTrial',
       'Decision', 'EmailType', 'PhaseValue', 'Confidence', 'EmailAction',
       'ReactionTime', 'Correct'],
      dtype='object')
"""

columns = ["Embedding Similarity", "User Outcome"]

correlation = pd.DataFrame([], columns=columns)


high_outcomes = []
low_outcomes = []
high_performances = []
low_performances = []
User_Outcomes = []
User_Mean_Performances = []

for idx, messageEmbedding in MessageEmbeddings.iterrows():
    UserAnnotations = Annotations[Annotations['UserId'] == messageEmbedding['UserId']]
    preTrainingAccuracy = UserAnnotations[UserAnnotations['PhaseValue'] == "preTraining"]['Correct'].mean()
    postTrainingAccuracy = UserAnnotations[UserAnnotations['PhaseValue'] == "postTraining"]['Correct'].mean()
    User_Outcome = int(round(((100 * (postTrainingAccuracy - preTrainingAccuracy)) / 10) * 10))

    if(User_Outcome > 0.2):
        high_outcomes.append(1)
    else:
        high_outcomes.append(0)
    
    if(User_Outcome < 0.2):
        low_outcomes.append(1)
    else: 
        low_outcomes.append(0)

    User_Outcomes.append(User_Outcome)

    User_Mean_Performance = UserAnnotations['Correct'].mean()

    if(User_Mean_Performance > 0.8):
        high_performances.append(1)
    else:
        high_performances.append(0)
    
    if(User_Outcome < 0.8):
        low_performances.append(1)
    else: 
        low_performances.append(0)

    User_Mean_Performances.append(User_Mean_Performance)

MessageEmbeddings['Low Outcome'] = low_outcomes
MessageEmbeddings["High Outcome"] = high_outcomes
MessageEmbeddings["User Outcome"] = User_Outcomes

MessageEmbeddings['Low Performance'] = low_performances
MessageEmbeddings["High Performance"] = high_performances
MessageEmbeddings["User Mean Performance"] = User_Mean_Performances

# Drop rows where either value is missing or non-numeric
MessageEmbeddings = MessageEmbeddings[
    MessageEmbeddings["High Outcome"].apply(lambda x: isinstance(x, (int, float))) &
    MessageEmbeddings["User Outcome"].apply(lambda x: isinstance(x, (int, float))) &
    MessageEmbeddings["Low Outcome"].apply(lambda x: isinstance(x, (int, float))) &
    MessageEmbeddings["User Mean Performance"].apply(lambda x: isinstance(x, (int, float)))
].dropna(subset=["Low Outcome", "High Outcome", "User Outcome", "User Mean Performance"])

# Select only the rows where 'High Outcome' is True
high_embeds = MessageEmbeddings.loc[MessageEmbeddings["High Outcome"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
high_embeds_stack = np.stack(high_embeds.values)
# Compute the mean embedding
mean_high_outcome_embeddings = high_embeds_stack.mean(axis=0).reshape(1, -1)

# Select only the rows where 'Low Outcome' is True
low_embeds = MessageEmbeddings.loc[MessageEmbeddings["Low Outcome"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
low_embeds_stack = np.stack(low_embeds.values)
# Compute the mean embedding
mean_low_outcome_embeddings = low_embeds_stack.mean(axis=0).reshape(1, -1)

# Select only the rows where 'High Performance' is True
high_embeds = MessageEmbeddings.loc[MessageEmbeddings["High Performance"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
high_embeds_stack = np.stack(high_embeds.values)
# Compute the mean embedding
mean_high_performance_embeddings = high_embeds_stack.mean(axis=0).reshape(1, -1)

# Select only the rows where 'Low Performance' is True
low_embeds = MessageEmbeddings.loc[MessageEmbeddings["Low Performance"] == 1, "Embedding"]
# If embeddings are stored as lists, convert to numpy array
low_embeds_stack = np.stack(low_embeds.values)
# Compute the mean embedding
mean_low_performance_embeddings = low_embeds_stack.mean(axis=0).reshape(1, -1)

columns = ["High Outcome Embedding Similarity", "Low Outcome Embedding Similarity",  "User Performance", "User Outcome"]
df = pd.DataFrame([], columns=columns)

for UserId in MessageEmbeddings['UserId'].unique():
    UserMessages = MessageEmbeddings[MessageEmbeddings['UserId'] == UserId]

    User_Embeddings = UserMessages["Embedding"]
    User_Embeddings_Stack = np.stack(User_Embeddings.values)
    User_Mean_Embedding = User_Embeddings_Stack.mean(axis=0).reshape(1, -1)

    High_Outcome_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_high_outcome_embeddings)[0][0]
    Low_Outcome_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_low_outcome_embeddings)[0][0]
    High_Performance_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_high_performance_embeddings)[0][0]
    Low_Performance_Embedding_Similarity = cosine_similarity(User_Mean_Embedding, mean_low_performance_embeddings)[0][0]
    User_Outcome = UserMessages['User Outcome'].unique()[0]
    User_Mean_Performance = UserMessages['User Mean Performance'].unique()[0]

    d = pd.DataFrame([{
                        "High Outcome Embedding Similarity":High_Outcome_Embedding_Similarity,
                        "Low Outcome Embedding Similarity":Low_Outcome_Embedding_Similarity,
                        "User Outcome":User_Outcome,
                        "User Performance":User_Mean_Performance,
                        "High Performance Embedding Similarity":High_Performance_Embedding_Similarity,
                        "Low Performance Embedding Similarity":Low_Performance_Embedding_Similarity,
                    }])
    
    if(len(df) == 0): 
        df = d
    else:
        df = pd.concat([df,d])


x_axis_2 = "High Outcome Embedding Similarity"
df = df[df[x_axis_2] > 0.925]

sns.jointplot(data=df, x="High Performance Embedding Similarity", y="High Outcome Embedding Similarity", hue="User Performance")

plt.show()

assert(False)
# Clean your data
cols = [
    "High Outcome Embedding Similarity",
    "Low Outcome Embedding Similarity",
    "High Performance Embedding Similarity",
    "Low Performance Embedding Similarity",
    "User Outcome",
    "User Performance"
]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df_clean = df[df[cols].applymap(np.isfinite).all(axis=1)]

# Plot all pairwise comparisons
sns.pairplot(df_clean[cols], diag_kind="kde")
plt.suptitle("Pairwise Relationships Between Embedding Similarities and User Metrics", y=1.02)
plt.show()

assert(False )
# Set up 3-column subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ensure all values are float
df["High Outcome Embedding Similarity"] = pd.to_numeric(df["High Outcome Embedding Similarity"], errors='coerce')
df["Low Outcome Embedding Similarity"] = pd.to_numeric(df["Low Outcome Embedding Similarity"], errors='coerce')
df["User Outcome"] = pd.to_numeric(df["User Outcome"], errors='coerce')
df["User Performance"] = pd.to_numeric(df["User Performance"], errors='coerce')

y_axis = "User Performance"
x_axis_1 = "High Performance Embedding Similarity"

df = df[df[x_axis_2] > 0.95]

# Plot High Outcome Embedding Similarity
sns.regplot(data=df, x=x_axis_1, y=y_axis, scatter=True, ci=95, ax=axes[0])
slope, intercept, r_value, p_value, std_err = linregress(df[x_axis_1], df[y_axis])
axes[0].text(0.05, 0.95, f'$R^2$ = {r_value**2:.2f}\np = {p_value:.4f}', transform=axes[0].transAxes, verticalalignment='top', fontsize=12)
axes[0].set_xlabel(x_axis_1)
axes[0].set_ylabel(y_axis)
axes[0].set_title("High Similarity vs. Performance")

# Plot Low Outcome Embedding Similarity
sns.regplot(data=df, x=x_axis_2, y=y_axis, scatter=True, ci=95, ax=axes[1])
slope, intercept, r_value, p_value, std_err = linregress(df[x_axis_2], df[y_axis])
axes[1].text(0.05, 0.95, f'$R^2$ = {r_value**2:.2f}\np = {p_value:.4f}', transform=axes[1].transAxes, verticalalignment='top', fontsize=12)
axes[1].set_xlabel(x_axis_2)
axes[1].set_ylabel(y_axis)
axes[1].set_title("Low Similarity vs. Performance")

# Plot distribution of User Outcome
sns.histplot(df["User Performance"].dropna(), kde=True, ax=axes[2])
axes[2].set_xlabel(y_axis)
axes[2].set_title("User Performance Distribution")

plt.tight_layout()
plt.show()