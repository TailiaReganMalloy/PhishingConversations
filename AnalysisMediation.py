import pandas as pd 
import numpy as np 
from Utils import *

DataFrame = pd.read_pickle("./Database/Merged.pkl")

Metrics = ['Age', 'Gender Number', 'Education Years', 'Phishing Experience', 'Chatbot Experience', 'AI Generation Perception', 'Pre Experiment Quiz Score', "Response Message Similarity", 'Cognitive Model Activity']
Roles = ['Student and Teacher', 'Teacher', 'Student']
Targets = ['Correct Categorization']
meddfs, mediations = plot_mediation(DataFrame=DataFrame, Roles=Roles, Targets=Targets, Metrics=Metrics)
for mediation in mediations: print(mediation)

#print(len(mediations))

