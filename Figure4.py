import pandas as pd 
from Utils import plot_helper

Merged = pd.read_pickle('./Database/Merged.pkl')

targets = ["User Initial Performance", "User Improvement",  "User Final Performance"]
(fig, plt) = plot_helper(Merged=Merged, targets=targets)


plt.show()
plt.savefig("Figures/Figure2.png")