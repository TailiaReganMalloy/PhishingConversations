import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity

def plot_helper(Merged, targets):
    # Plot regression using binned means
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax_idx, (ax, target) in enumerate(zip(axes.flat, targets)):
        sub = Merged[["Message Email Similarity", 'Role', target]].copy()
        # ensure numeric y
        sub[target] = pd.to_numeric(sub[target], errors="coerce")
        
        sub = sub[sub[target].notna() & sub["Message Email Similarity"].notna()]
        sub["_role"] = sub['Role'].astype(str).str.lower()


        # Bin similarity by 0.01 (round to 2 decimals) and aggregate per bin+role
        sub["SimBin"] = sub["Message Email Similarity"].round(2)
        agg = (
            sub.groupby(["_role", "SimBin"], as_index=False)
            .agg(
                x=("Message Email Similarity", "mean"),  # bin mean of similarity
                y=(target, "mean"),                                  # bin mean of target
                n=("Message Email Similarity", "size"),   # bin count (for reference)
            )
        )

        ax.set_xlim([0,1])

        for role_name in ["user", "system"]:
            ss = agg[agg["_role"] == role_name]
            role_label = "Human Student" if role_name == "user" else "Teacher LLM"
            if ss.empty:
                continue
            sns.regplot(
                data=ss,
                x="x",
                y="y",
                ax=ax,
                scatter=True,
                ci=95,
                scatter_kws={"alpha": 0.5, "s": 30},
                label=role_label,
                truncate=False
            )
            # --- Add regression stats as text ---
            # Only compute stats if there are at least two unique x points
            if len(ss) >= 2 and ss["x"].nunique() >= 2:
                res = linregress(ss["x"], ss["y"])  # slope, intercept, rvalue, pvalue, etc.
                r2 = res.rvalue ** 2
                pval = res.pvalue

                # Stack annotations near the top of each axes; one line per role
                y_text = 0.98 if role_name == "user" else 0.90
                ax.text(
                    0.02,
                    y_text,
                    f"{role_label}: $R^2$={r2:.3f}, p={pval:.3g}",
                    transform=ax.transAxes,
                    fontsize=14,
                    va="top",
                )

        if(ax_idx == 0 ):
            ax.legend(title="Message Sender", loc='lower left', title_fontsize=16, fontsize=14)
            ax.set_ylim((0.2,1.2))
        else: 
            legend = ax.legend()
            legend.remove()

        ax.set_title(f"{target} vs Message↔Email\nCosine Similarity", fontsize=18)
        ax.set_xlabel("Message↔Email Cosine Similarity", fontsize=16)
        ax.set_ylabel(f"{target}", fontsize=16)
    
    return (fig, plt) 