import pandas as pd 
from Utils import plot_regression


Merged = pd.read_pickle('./Database/Merged.pkl')

Targets = ["Correct Categorization", "Categorization Confidence", "Reaction Time"]
(fig, plt) = plot_regression(DataFrame=Merged, Targets=Targets)


plt.show()
plt.savefig("Figures/Figure1.png")
