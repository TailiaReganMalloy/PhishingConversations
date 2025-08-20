"""

"""

import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity


MessageEmbeddings = pd.read_pickle('./Database/MessageEmbeddings.pkl')
"""print(MessageEmbeddings.columns)
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct', 'Role', 'Content', 'MessageEmbedding'],
      dtype='object')
"""
MessageEmbeddings.rename(columns={'Embedding':'MessageEmbedding'}, inplace=True)
MessageEmbeddings["MessageEmbedding"] = [np.array([float(y) for y in x]).reshape(1,-1) for x in MessageEmbeddings["MessageEmbedding"]]

Annotations =  pd.read_pickle('./Database/Annotations.pkl')
"""
Index(['UserId', 'Experiment', 'ExperimentCondition', 'EmailId', 'PhaseTrial',
       'Decision', 'EmailType', 'PhaseValue', 'Confidence', 'EmailAction',
       'ReactionTime', 'Correct'],
      dtype='object')
"""

Demographics = pd.read_pickle('./Database/Demographics.pkl')
"""
Index(['UserId', 'Age', 'Gender', 'Education', 'Country', 'Victim', 'Chatbot',
       'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5'],
      dtype='object')
"""

EmailEmbeddings = pd.read_pickle("./Database/EmailEmbeddings.pkl")
"""print(EmailEmbeddings.columns)
Index(['EmailId', 'BaseEmailID', 'Author', 'Style', 'Embedding',
       'Phishing Similarity', 'Ham Similarity'],
      dtype='object')
"""
EmailEmbeddings.rename(columns={'Embedding':'EmailEmbedding'}, inplace=True)
EmailEmbeddings["EmailEmbedding"] = [np.array([float(y) for y in x]).reshape(1,-1) for x in EmailEmbeddings["EmailEmbedding"]]

Improvements = []

MessageEmbeddings['User Initial Performance'] = [Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'preTraining')]['Correct'].mean() for _, MessageEmbedding in MessageEmbeddings.iterrows()]

MessageEmbeddings['User Improvement'] = [Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'postTraining')]['Correct'].mean() - Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'preTraining')]['Correct'].mean() for _, MessageEmbedding in MessageEmbeddings.iterrows()]

MessageEmbeddings['User Final Performance'] = [Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'postTraining')]['Correct'].mean() for _, MessageEmbedding in MessageEmbeddings.iterrows()]


for idx, EmailEmbedding in EmailEmbeddings.iterrows():
    if(EmailEmbedding['EmailEmbedding'].shape != (1,3072)):
        print(EmailEmbedding['EmailEmbedding'].shape)
        assert(False)


# ── Join message → email embeddings ───────────────────────────────────────────
df = MessageEmbeddings.merge(
    EmailEmbeddings[["EmailId", "EmailEmbedding"]],
    on="EmailId",
    how="left",
    suffixes=("", "_Email"),
)
"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct', 'Role', 'Content', 'MessageEmbedding', 'User Improvement',
       'AI Writing Perception', 'AI Code Generation Perception',
       'EmailEmbedding'],
      dtype='object')
"""

# Keep only valid, same-dimension pairs
df = df.dropna(subset=["MessageEmbedding", "EmailEmbedding"]).copy()
same_dim = df.apply(lambda r: r["MessageEmbedding"].shape[1] == r["EmailEmbedding"].shape[1], axis=1)
df = df[same_dim].copy()

# Compute scalar cosine similarity
df["Message Cosine Similarity to Email"] = [
    float(round(float(cosine_similarity(m, e)[0, 0]) / 0.025) * 0.025) for m, e in zip(df["MessageEmbedding"], df["EmailEmbedding"])
]

df['Message Cosine Similarity to Email'] = df.groupby('Role')['Message Cosine Similarity to Email'].transform(
    lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
)
# Normalize to between 0 and 1 
# ── Plots: similarity vs outcomes, split by role ─────────────────────
targets = ["User Initial Performance", "User Improvement",  "User Final Performance"]
role_col = 'Role'

# Plot regression using binned means
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
for ax_idx, (ax, target) in enumerate(zip(axes.flat, targets)):
    sub = df[["Message Cosine Similarity to Email", role_col, target]].copy()
    # ensure numeric y
    sub[target] = pd.to_numeric(sub[target], errors="coerce")
    
    sub = sub[sub[target].notna() & sub["Message Cosine Similarity to Email"].notna()]
    sub["_role"] = sub[role_col].astype(str).str.lower()


    # Bin similarity by 0.01 (round to 2 decimals) and aggregate per bin+role
    sub["SimBin"] = sub["Message Cosine Similarity to Email"].round(2)
    agg = (
        sub.groupby(["_role", "SimBin"], as_index=False)
        .agg(
            x=("Message Cosine Similarity to Email", "mean"),  # bin mean of similarity
            y=(target, "mean"),                                  # bin mean of target
            n=("Message Cosine Similarity to Email", "size"),   # bin count (for reference)
        )
    )

    ax.set_xlim([0,1])

    for role_name in ["user", "system"]:
        ss = agg[agg["_role"] == role_name]
        role_label = "Human Student" if role_name == "user" else "Teacher LLM"
        if ss.empty:
            continue
        sns.regplot(
            data=ss,
            x="x",
            y="y",
            ax=ax,
            scatter=True,
            ci=95,
            scatter_kws={"alpha": 0.5, "s": 30},
            label=role_label,
            truncate=False
        )
        # --- Add regression stats as text ---
        # Only compute stats if there are at least two unique x points
        if len(ss) >= 2 and ss["x"].nunique() >= 2:
            res = linregress(ss["x"], ss["y"])  # slope, intercept, rvalue, pvalue, etc.
            r2 = res.rvalue ** 2
            pval = res.pvalue

            # Stack annotations near the top of each axes; one line per role
            y_text = 0.98 if role_name == "user" else 0.90
            ax.text(
                0.02,
                y_text,
                f"{role_label}: $R^2$={r2:.3f}, p={pval:.3g}",
                transform=ax.transAxes,
                fontsize=14,
                va="top",
            )

    if(ax_idx == 0 ):
        ax.legend(title="Message Sender", loc='lower left', title_fontsize=16, fontsize=14)
    else: 
        legend = ax.legend()
        legend.remove()

    ax.set_title(f"{target}\nvs Message↔Email Cosine Similarity", fontsize=16)
    ax.set_xlabel("Message Cosine Similarity to Email", fontsize=14)
    ax.set_ylabel(target, fontsize=14)

plt.show()
