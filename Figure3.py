import pandas as pd 
from Utils import plot_helper

Merged = pd.read_pickle('./Database/Merged.pkl')
targets = ["Pre Experiment Quiz Score", "AI Generation Perception", "Response Message Similarity"]
(fig, plt) = plot_helper(Merged=Merged, targets=targets)

plt.show()
plt.savefig("Figures/Figure3.png")