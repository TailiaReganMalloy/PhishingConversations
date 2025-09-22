import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy 

from Utils import compare_embeddings, plot_msg_vs_correct_binned

emails = pd.read_pickle("./Database/Emails.pkl")
orig = pd.read_pickle("./Database/Messages.pkl")

comparisons, msg_features = compare_embeddings(emails, orig)

comparisons.to_pickle("./Database/Comparisons.pkl")
msg_features.to_pickle("./Database/Message_Features.pkl")

# ------------------------
# Plot layout: top row full-width (left graph), bottom row with 3 panels
# Use constrained_layout and remove vertical spacing between rows
# ------------------------
fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.5, 1.0], hspace=0.0, wspace=0.0)

ax_top = fig.add_subplot(gs[0, :])   # top row spanning all 3 columns
# bottom row: share y-axis so only left shows y ticks/label
ax_bot1 = fig.add_subplot(gs[1, 0])
ax_bot2 = fig.add_subplot(gs[1, 1], sharey=ax_bot1)
ax_bot3 = fig.add_subplot(gs[1, 2], sharey=ax_bot1)

# compute mean across the three per-model r values and plot mean-only
comparisons['r_mean'] = comparisons[['r_correct', 'r_reaction', 'r_confidence']].mean(axis=1)
metric = 'r_mean'
pmetric = 'p_correct'

# regression line (Embedding Size on x, mean r on y)
sns.regplot(data=comparisons, x="Embedding Size", y=metric, ci=95, scatter=False, line_kws={"color":"C0"}, ax=ax_top)
# base points (fixed marker)
ax_top.scatter(comparisons["Embedding Size"], comparisons[metric], color="C1", alpha=0.9, s=80, zorder=5)

# overlay composite as sized markers (if available) - mean only (no std)
comp_df = comparisons.dropna(subset=['composite_metric', metric, 'Embedding Size']).copy()
if not comp_df.empty:
    # map composite (0..1) -> marker size and a colormap)
    cmap = plt.cm.get_cmap("viridis")
    sc = ax_top.scatter(comp_df['Embedding Size'], comp_df[metric], c=comp_df['composite_metric'], cmap=cmap,
                        edgecolors='k', linewidth=0.6, zorder=6, label="composite (msg_len/prop_common/ngrams)")
    cbar = fig.colorbar(sc, ax=ax_top, pad=0.02)
    cbar.set_label("Composite (normalized)")

# overall pearson between mean r and Embedding Size
if len(comparisons) >= 2:
    overall_r, overall_p = scipy.stats.pearsonr(comparisons[metric].dropna(), comparisons["Embedding Size"].dropna())
    ax_top.text(0.02, 0.95, f"Overall $r^2$={overall_r:.3f}, p={overall_p:.2g}", transform=ax_top.transAxes, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.75), fontsize=12)

# annotate each model label near its primary point
below = ['BAAI BGE-Large', 'OpenAI\nSmall 3']
for _, row in comparisons.dropna(subset=[metric, 'Embedding Size']).iterrows():
    r_val = row[metric]
    p_val = row.get(pmetric, np.nan)
    label = f"{row['model']}\nmean $r^2$={r_val:.3f}\np={p_val:.2g}"
    # place specific models' labels below the point, others above-right
    if row['model'] in below:
        ax_top.annotate(label, (row["Embedding Size"], r_val), textcoords="offset points", xytext=(6,-14), va="top", fontsize=8)
    else:
        ax_top.annotate(label, (row["Embedding Size"], r_val), textcoords="offset points", xytext=(6,6), va="bottom", fontsize=8)
ax_top.set_title("Cosine Similarity Correlation to Learning Metrics By Model Embedding Size")
ax_top.set_xlabel("Embedding Size")
ax_top.set_ylabel(" Correlation to Learning Metrics")
#ax_top.set_yscale('log')
#ax_top.set_xscale('log')
#ax_top.set_ylim(5,10)

# Bottom panels
plot_msg_vs_correct_binned(msg_features, ax_bot1, 'msg_length', 'Message length (tokens)')
ax_bot1.set_title("Message length vs\nLearning Metrics")
ax_bot1.set_ylabel("Normalized Learning Metric")
ax_bot1.set_ylim(0.6,1.1)

plot_msg_vs_correct_binned(msg_features, ax_bot2, 'prop_common', 'Proportion common words')
ax_bot2.set_title("Prop. common words vs\nLearning Metrics")
ax_bot2.set_yticklabels([])
ax_bot2.set_ylim(0.6,1.1)

plot_msg_vs_correct_binned(msg_features, ax_bot3, 'ngram_counts', 'Shared n-gram count (n=3 or 4)')
ax_bot3.set_title("Shared n-grams vs\nLearning Metrics")
ax_bot3.set_yticklabels([])
#ax_bot3.set_xlim(0,8)
ax_bot3.set_ylim(0.6,1.1)

plt.show()

