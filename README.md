# PhishingConversations

The data in the database folder is available in osf: https://osf.io/wbg3r/

Additional datasets of phishing emails and human behavior are available at: https://osf.io/r83ag/ and https://osf.io/r83ag/
# Database Folder 

See the readme folder for information on each of the files in this folder and the meaning of all of the column names. 
To load pickle files import pandas and run the following code:

```python 
import pandas as pd 

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
```

# Previous Papers Folder 

## CHB2024 Computers in Human Behavior paper code. 

Code for the paper in press for CHB, see the folder for the manuscript. 

## Conference on Natural Language Learning 

Code for the CoNLL paper 

## Frontiers in Psychology

Code for the Frontiers in Psychology paper. 