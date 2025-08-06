import numpy as np 
import pandas as pd 
import seaborn as sns 

Emails = pd.read_pickle('./Database/Emails.pkl')

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

