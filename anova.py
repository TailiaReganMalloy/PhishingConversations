import pandas as pd
import pingouin as pg 
import numpy as np 

from sklearn.metrics.pairwise import cosine_similarity

MergedEmbeddingDictionary = pd.read_pickle("./Database/MergedEmeddingDictionary.pkl")
"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct', 'Role', 'Content', 'MessageEmbedding', 'User Improvement',
       'Perception of Emails as AI Generated', 'Pre-Experiment Quiz Score',
       'Response Message Similarity', 'EmailEmbedding', 'Age', 'Gender',
       'Education', 'Country', 'Victim', 'Chatbot', 'Q0', 'Q1', 'Q2', 'Q3',
       'Q4', 'Q5', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5', 'DemographicsEmbedding'],
      dtype='object'
"""

Annotations =  pd.read_pickle('./Database/Annotations.pkl')
"""
Index(['UserId', 'Experiment', 'ExperimentCondition', 'EmailId', 'PhaseTrial',
       'Decision', 'EmailType', 'PhaseValue', 'Confidence', 'EmailAction',
       'ReactionTime', 'Correct'],
      dtype='object')
"""

# Compute scalar cosine similarity
MergedEmbeddingDictionary["Message Cosine Similarity to Email"] = [
    float(round(float(cosine_similarity(m, e)[0, 0]))) for m, e in zip(MergedEmbeddingDictionary["MessageEmbedding"], MergedEmbeddingDictionary["EmailEmbedding"])
]

MergedEmbeddingDictionary['Message Cosine Similarity to Email'] = MergedEmbeddingDictionary.groupby('Role')['Message Cosine Similarity to Email'].transform(
    lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
)

MergedEmbeddingDictionary['User Initial Performance'] = [Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'preTraining')]['Correct'].mean() for _, MessageEmbedding in MergedEmbeddingDictionary.iterrows()]

MergedEmbeddingDictionary['User Improvement'] = [Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'postTraining')]['Correct'].mean() - Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'preTraining')]['Correct'].mean() for _, MessageEmbedding in MergedEmbeddingDictionary.iterrows()]

MergedEmbeddingDictionary['User Final Performance'] = [Annotations[(Annotations['UserId'] == MessageEmbedding['UserId']) & (Annotations['PhaseValue'] == 'postTraining')]['Correct'].mean() for _, MessageEmbedding in MergedEmbeddingDictionary.iterrows()]

# ── Plots: similarity vs outcomes, split by role ─────────────────────
targets = ['Age', 'Gender', 'Education', 'Victim', 'Chatbot', 'ExperimentCondition']
# 1) Binary / ordinal
MergedEmbeddingDictionary['Gender Number'] = MergedEmbeddingDictionary['Gender'].map({'M': 0, 'F': 1}).astype('float')

# Use an *ordered* scale for education (years is nicer than 0..3)
edu_map = {'HS': 12, 'BD': 16, 'MD': 18, 'PD': 21}
MergedEmbeddingDictionary['Education Years'] = MergedEmbeddingDictionary['Education'].map(edu_map).astype('float')

exp_map = {np.nan: 0, 'A': 1, 'YM': 2, 'YF': 3}
MergedEmbeddingDictionary['Phishing Experience'] = MergedEmbeddingDictionary['Victim'].map(exp_map).astype('float')

exp_map = {np.nan: 0, 'A': 1, 'YM': 2, 'YF': 3}
MergedEmbeddingDictionary['Chatbot Experience'] = MergedEmbeddingDictionary['Chatbot'].map(exp_map).astype('float')

exp_map = {'IBL Emails Written Feedback':2, 'IBL Emails Points Feedback':1, 'Random Emails Written Feedback':1, 'Ablation Experiment':0}
MergedEmbeddingDictionary['Cognitive Model Activity'] = MergedEmbeddingDictionary['ExperimentCondition'].map(exp_map).astype('float')

MergedEmbeddingDictionary.to_pickle("Database/MergedEmbeddingDictionary.pkl")
MergedEmbeddingDictionary.to_csv("Database/MergedEmbeddingDictionary.csv")

# Compute scalar cosine similarity
MergedEmbeddingDictionary["Message Cosine Similarity to Email"] = [
    float(round(float(cosine_similarity(m, e)[0, 0]) / 0.025) * 0.025) for m, e in zip(MergedEmbeddingDictionary["MessageEmbedding"], MergedEmbeddingDictionary["EmailEmbedding"])
]

## ANOVA requires binned continuous values 
MergedEmbeddingDictionary['Message Cosine Similarity to Email'] = MergedEmbeddingDictionary.groupby('Role')['Message Cosine Similarity to Email'].transform(
    lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
)



"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct', 'Role', 'Content', 'MessageEmbedding', 'User Improvement',
       'Perception of Emails as AI Generated', 'Pre-Experiment Quiz Score',
       'Response Message Similarity', 'EmailEmbedding', 'Age', 'Gender',
       'Education', 'Country', 'Victim', 'Chatbot', 'Q0', 'Q1', 'Q2', 'Q3',
       'Q4', 'Q5', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5', 'DemographicsEmbedding',
       'Message Cosine Similarity to Email', 'User Initial Performance',
       'User Final Performance', 'Gender Number', 'Education Years',
       'Phishing Experience', 'Chatbot Experience',
       'Cognitive Model Activity'],
      dtype='object')
"""

def latex_anova_line(a, label="ANOVA"):
    """Format a pingouin anova()/welch_anova() result as LaTeX."""
    row = a.iloc[0] if len(a) else {}
    F = float(row.get("F", np.nan))
    p = float(row.get("p-unc", row.get("p-GG", row.get("p-HF", np.nan))))
    np2 = row.get("np2", np.nan)

    df1 = row.get("ddof1", None)
    df2 = row.get("ddof2", None)

    def _fmt_df(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "?"
        # show integer if close to int; otherwise 2 decimals (Welch can be fractional)
        return f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:.2f}"

    parts = [f"{label}: $F({ _fmt_df(df1) }, { _fmt_df(df2) })={F:.3f}$",
             f"$p={p:.3g}$"]
    if not (isinstance(np2, float) and np.isnan(np2)):
        parts.append(f"$\\eta_p^2={float(np2):.3f}$")
    # keep the BF10 placeholder for consistency with your t-test line
    parts.append("BF10=NA")
    return ", ".join(parts)

# Print the t-test statistics for each regression in the order they appear in the manuscript
columns = ['Correct', 'Confidence', 'ReactionTime', 'User Initial Performance', 'User Improvement', 'User Final Performance', 'Pre-Experiment Quiz Score', 'Perception of Emails as AI Generated', 'Response Message Similarity', 'Age', 'Gender Number', 'Education Years', 'Phishing Experience', 'Chatbot Experience', 'Cognitive Model Activity']
for column in columns: 
    studentMessages = MergedEmbeddingDictionary[MergedEmbeddingDictionary['Role'] == 'user']
    anova = pg.anova(data=studentMessages, dv=column, between='Message Cosine Similarity to Email')
    print("Student Message-Email ECSS vs. " + column + " ANOVA ")
    print(latex_anova_line(anova))

    teacherMessages = MergedEmbeddingDictionary[MergedEmbeddingDictionary['Role'] == 'system']
    anova = pg.anova(data=teacherMessages, dv=column, between='Message Cosine Similarity to Email')
    print("Teacher Message-Email ECSS " + column + " ANOVA")
    print(latex_anova_line(anova))

"""
Student Message-Email ECSS vs. Correct ANOVA 
ANOVA: $F(22, 464)=0.841$, $p=0.674$, $\eta_p^2=0.038$
Teacher Message-Email ECSS Correct ANOVA
ANOVA: $F(25, 1720)=1.648$, $p=0.0231$, $\eta_p^2=0.023$
Student Message-Email ECSS vs. Confidence ANOVA 
ANOVA: $F(22, 464)=1.539$, $p=0.0569$, $\eta_p^2=0.068$
Teacher Message-Email ECSS Confidence ANOVA
ANOVA: $F(25, 1720)=1.652$, $p=0.0225$, $\eta_p^2=0.023$
Student Message-Email ECSS vs. ReactionTime ANOVA 
ANOVA: $F(22, 464)=1.155$, $p=0.284$, $\eta_p^2=0.052$
Teacher Message-Email ECSS ReactionTime ANOVA
ANOVA: $F(25, 1720)=0.882$, $p=0.632$, $\eta_p^2=0.013$
Student Message-Email ECSS vs. User Initial Performance ANOVA 
ANOVA: $F(22, 464)=0.692$, $p=0.849$, $\eta_p^2=0.032$
Teacher Message-Email ECSS User Initial Performance ANOVA
ANOVA: $F(25, 1720)=0.863$, $p=0.659$, $\eta_p^2=0.012$
Student Message-Email ECSS vs. User Improvement ANOVA 
ANOVA: $F(22, 464)=1.557$, $p=0.0521$, $\eta_p^2=0.069$
Teacher Message-Email ECSS User Improvement ANOVA
ANOVA: $F(25, 1720)=1.014$, $p=0.444$, $\eta_p^2=0.015$
Student Message-Email ECSS vs. User Final Performance ANOVA 
ANOVA: $F(22, 464)=1.705$, $p=0.0247$, $\eta_p^2=0.075$
Teacher Message-Email ECSS User Final Performance ANOVA
ANOVA: $F(25, 1720)=1.189$, $p=0.237$, $\eta_p^2=0.017$
Student Message-Email ECSS vs. Pre-Experiment Quiz Score ANOVA 
ANOVA: $F(22, 464)=1.195$, $p=0.247$, $\eta_p^2=0.054$
Teacher Message-Email ECSS Pre-Experiment Quiz Score ANOVA
ANOVA: $F(25, 1720)=1.261$, $p=0.174$, $\eta_p^2=0.018$
Student Message-Email ECSS vs. Perception of Emails as AI Generated ANOVA 
ANOVA: $F(22, 464)=1.348$, $p=0.135$, $\eta_p^2=0.060$
Teacher Message-Email ECSS Perception of Emails as AI Generated ANOVA
ANOVA: $F(25, 1720)=0.702$, $p=0.86$, $\eta_p^2=0.010$
Student Message-Email ECSS vs. Response Message Similarity ANOVA 
ANOVA: $F(22, 464)=5.624$, $p=4.86e-14$, $\eta_p^2=0.211$
Teacher Message-Email ECSS Response Message Similarity ANOVA
ANOVA: $F(25, 1720)=1.377$, $p=0.102$, $\eta_p^2=0.020$
Student Message-Email ECSS vs. Age ANOVA 
ANOVA: $F(22, 464)=1.395$, $p=0.11$, $\eta_p^2=0.062$
Teacher Message-Email ECSS Age ANOVA
ANOVA: $F(25, 1720)=1.122$, $p=0.307$, $\eta_p^2=0.016$
Student Message-Email ECSS vs. Gender Number ANOVA 
ANOVA: $F(22, 464)=1.110$, $p=0.331$, $\eta_p^2=0.050$
Teacher Message-Email ECSS Gender Number ANOVA
ANOVA: $F(25, 1720)=0.880$, $p=0.635$, $\eta_p^2=0.013$
Student Message-Email ECSS vs. Education Years ANOVA 
ANOVA: $F(22, 464)=0.991$, $p=0.474$, $\eta_p^2=0.045$
Teacher Message-Email ECSS Education Years ANOVA
ANOVA: $F(25, 1720)=0.984$, $p=0.486$, $\eta_p^2=0.014$
Student Message-Email ECSS vs. Phishing Experience ANOVA 
ANOVA: $F(22, 464)=0.923$, $p=0.565$, $\eta_p^2=0.042$
Teacher Message-Email ECSS Phishing Experience ANOVA
ANOVA: $F(25, 1720)=0.912$, $p=0.589$, $\eta_p^2=0.013$
Student Message-Email ECSS vs. Chatbot Experience ANOVA 
ANOVA: $F(22, 464)=1.332$, $p=0.144$, $\eta_p^2=0.059$
Teacher Message-Email ECSS Chatbot Experience ANOVA
ANOVA: $F(25, 1720)=1.016$, $p=0.442$, $\eta_p^2=0.015$
Student Message-Email ECSS vs. Cognitive Model Activity ANOVA 
ANOVA: $F(22, 464)=1.725$, $p=0.0222$, $\eta_p^2=0.076$
Teacher Message-Email ECSS Cognitive Model Activity ANOVA
ANOVA: $F(25, 1720)=1.159$, $p=0.267$, $\eta_p^2=0.017$

"""


"""
T-Tests
Student Message-Email ECSS vs. Correct T-Test
, T-Test: $T=-12.948888, p=1.7e-35, CI95\%=[-0.33, -0.24]$
Teacher Message-Email ECSS Correct T-Test
, T-Test: $T=-40.340361, p=2.1e-292, CI95\%=[-0.41, -0.37]$
Student Message-Email ECSS vs. Confidence T-Test
, T-Test: $T=-48.477282, p=1.2e-261, CI95\%=[-2.50, -2.31]$
Teacher Message-Email ECSS Confidence T-Test
, T-Test: $T=-107.219555, p=0.0e+00, CI95\%=[-2.65, -2.55], BF10=\infty$
Student Message-Email ECSS vs. ReactionTime T-Test
, T-Test: $T=-29.767941, p=6.4e-139, CI95\%=[-15880.73, -13916.40]$
Teacher Message-Email ECSS ReactionTime T-Test
, T-Test: $T=-52.143537, p=0.0e+00, CI95\%=[-16992.68, -15761.11], BF10=\infty$
Student Message-Email ECSS vs. User Initial Performance T-Test
, T-Test: $T=-26.754198, p=1.3e-118, CI95\%=[-0.32, -0.28]$
Teacher Message-Email ECSS User Initial Performance T-Test
, T-Test: $T=-67.851027, p=0.0e+00, CI95\%=[-0.36, -0.34], BF10=\infty$
Student Message-Email ECSS vs. User Improvement T-Test
, T-Test: $T=34.584272, p=1.6e-171, CI95\%=[0.40, 0.44]$
Teacher Message-Email ECSS User Improvement T-Test
, T-Test: $T=74.486671, p=0.0e+00, CI95\%=[0.38, 0.41], BF10=\infty$
Student Message-Email ECSS vs. User Final Performance T-Test
, T-Test: $T=-25.191985, p=3.5e-108, CI95\%=[-0.34, -0.29]$
Teacher Message-Email ECSS User Final Performance T-Test
, T-Test: $T=-76.470812, p=0.0e+00, CI95\%=[-0.40, -0.38], BF10=\infty$
Student Message-Email ECSS vs. Pre-Experiment Quiz Score T-Test
, T-Test: $T=-32.279605, p=6.1e-156, CI95\%=[-1.11, -0.98]$
Teacher Message-Email ECSS Pre-Experiment Quiz Score T-Test
, T-Test: $T=-63.248801, p=0.0e+00, CI95\%=[-1.17, -1.10], BF10=\infty$
Student Message-Email ECSS vs. Perception of Emails as AI Generated T-Test
, T-Test: $T=-61.210063, p=0.0e+00, CI95\%=[-62.48, -58.60], BF10=\infty$
Teacher Message-Email ECSS Perception of Emails as AI Generated T-Test
, T-Test: $T=-114.778277, p=0.0e+00, CI95\%=[-56.49, -54.59], BF10=\infty$
Student Message-Email ECSS vs. Response Message Similarity T-Test
, T-Test: $T=15.846432, p=1.8e-50, CI95\%=[0.14, 0.18]$
Teacher Message-Email ECSS Response Message Similarity T-Test
, T-Test: $T=7.440460, p=1.3e-13, CI95\%=[0.02, 0.04]$
Student Message-Email ECSS vs. Age T-Test
, T-Test: $T=-85.276217, p=0.0e+00, CI95\%=[-41.44, -39.58], BF10=\infty$
Teacher Message-Email ECSS Age T-Test
, T-Test: $T=-165.746470, p=0.0e+00, CI95\%=[-40.73, -39.78], BF10=\infty$
Student Message-Email ECSS vs. Gender Number T-Test
, T-Test: $T=0.238267, p=0.812, CI95\%=[-0.04, 0.05]$
Teacher Message-Email ECSS Gender Number T-Test
, T-Test: $T=4.792790, p=1.7e-06, CI95\%=[0.03, 0.08]$
Student Message-Email ECSS vs. Education Years T-Test
, T-Test: $T=-139.608352, p=0.0e+00, CI95\%=[-15.27, -14.85], BF10=\infty$
Teacher Message-Email ECSS Education Years T-Test
, T-Test: $T=-268.615390, p=0.0e+00, CI95\%=[-14.70, -14.49], BF10=\infty$
Student Message-Email ECSS vs. Phishing Experience T-Test
, T-Test: $T=-42.776738, p=1.1e-225, CI95\%=[-1.79, -1.64]$
Teacher Message-Email ECSS Phishing Experience T-Test
, T-Test: $T=-119.399339, p=0.0e+00, CI95\%=[-1.95, -1.88], BF10=\infty$
Student Message-Email ECSS vs. Chatbot Experience T-Test
, T-Test: $T=-75.615717, p=0.0e+00, CI95\%=[-2.19, -2.08], BF10=\infty$
Teacher Message-Email ECSS Chatbot Experience T-Test
, T-Test: $T=-138.964120, p=0.0e+00, CI95\%=[-2.14, -2.08], BF10=\infty$
Student Message-Email ECSS vs. Cognitive Model Activity T-Test
, T-Test: $T=-15.087086, p=2.3e-46, CI95\%=[-0.58, -0.45]$
Teacher Message-Email ECSS Cognitive Model Activity T-Test
, T-Test: $T=-37.292329, p=1.7e-256, CI95\%=[-0.76, -0.69]$
"""