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

