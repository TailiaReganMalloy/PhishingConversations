import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings('ignore')

# Load data
MessageEmbeddings = pd.read_pickle('./Database/MessageEmbeddings.pkl')
"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Email Similarity', 'Role', 'Content', 'Embedding', 'Low Improvement',
       'High Improvement', 'User Improvement', 'Low Performance',
       'High Performance', 'User Performance', 'Email Similarity'],
      dtype='object')
"""

# Calculate centroid embeddings per user
user_centroids = MessageEmbeddings.groupby('UserId')['Embedding'].apply(lambda x: np.mean(np.stack(x), axis=0))
user_centroids = pd.DataFrame(user_centroids.tolist(), index=user_centroids.index)

# Standardize embeddings before clustering
scaler = StandardScaler()
scaled_centroids = scaler.fit_transform(user_centroids)

# Cluster embeddings into 3 groups (low, medium, high performance clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
user_centroids['Cluster'] = kmeans.fit_predict(scaled_centroids)

"""
Low Performance Low Improvement 
Low Performance High Improvement 
High Performance Low Improvement 
High Performance High Improvement 
"""

"""
0: Low-High 
1: High-High
2: Low-Low
3: High-Low 
"""

# Merge cluster information back into the original dataset
performance_df = MessageEmbeddings.groupby('UserId').agg({
    'Email_Similarity': 'mean',
    'User_Improvement': 'mean',
    'User_Performance': 'mean',
    'ExperimentCondition': 'first'
})

performance_df = performance_df.merge(user_centroids[['Cluster']], left_index=True, right_index=True)

# Plot performance metrics by cluster
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.boxplot(data=performance_df, x='Cluster', y='Email_Similarity', ax=axes[0])
axes[0].set_title('User Email Similarity by Embedding Cluster')

sns.boxplot(data=performance_df, x='Cluster', y='User_Improvement', ax=axes[1])
axes[1].set_title('User Improvement by Embedding Cluster')

sns.boxplot(data=performance_df, x='Cluster', y='User_Performance', ax=axes[2])
axes[2].set_title('User Performance by Embedding Cluster')

plt.tight_layout()
plt.show()


# ANOVA test for Accuracy by Cluster
model = ols('Email Similarity ~ C(Cluster)', data=performance_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA results for Email Similarity by Cluster:\n", anova_table)

# Additional analysis: Cluster vs ExperimentCondition
cluster_exp_cond = pd.crosstab(performance_df['Cluster'], performance_df['ExperimentCondition'])
print("\nCluster vs Experiment Condition Crosstab:\n", cluster_exp_cond)