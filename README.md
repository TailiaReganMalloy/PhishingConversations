# ToDo list 
1. (DONE) Get embeddings for the conversations, add them to a new pkl file. 
2. Compare the embeddings for conversations and pariticpant learning outcomes. 
3. See if the difficulty of emails can be predicted from embeddings. 
4. How could we improve phishing education? 
5. What interesting information is there within the embeddings of GPT generated HTML/CSS code? 

# Main Analysis To Do 

If you would like to help with this, please take a look at the Database/MessageEmbeddings file, the version uploaded to gitHub is zipped since the whole file is above 100MB. Generally we would like to compare the conversations that people are having with the LLM chatbot and some other information to see if there is a correlation between embeddings and this other information. This other information can include things like participant demographics (age, gender, etc.) or their learning outcomes (training improvement), or the emails that they are looking at. 

# generateEmbeddings.py File 

Azure OpenAI Message Embedding Pipeline

This script loads phishing experiment dataframes, cleans/parses chat messages, 
extracts message roles/contents, and computes message embeddings using the Azure OpenAI Embeddings API.  
The processed dataframe is saved with new 'Role', 'Content', and 'Embedding' columns as both a pickle and CSV file.

Key workflow:
- Load all relevant pickled dataframes.
- Clean and parse each message for JSON content; extract roles and messages.
- Define an OpenAI embedder using Azure API (with retry logic for rate limits).
- Compute an embedding for each cleaned message (tokenized/truncated as needed).
- Save the enriched Messages dataframe for downstream use.

Dependencies:
- pandas, tqdm, tiktoken, numpy, dotenv, openai[azure], azure-identity, python-dotenv
- Requires Azure OpenAI environment variables in .env (endpoint, key, deployment, etc.)


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

