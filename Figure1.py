import pandas as pd 
from Utils import plot_helper


Merged = pd.read_pickle('./Database/Merged.pkl')

targets = ["Correct Categorization", "Categorization Confidence", "Reaction Time"]
(fig, plt) = plot_helper(Merged=Merged, targets=targets)


plt.show()
plt.savefig("Figures/Figure1.png")
