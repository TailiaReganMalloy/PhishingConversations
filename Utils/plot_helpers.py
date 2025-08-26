import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import pandas as pd 
import seaborn as sns 
import pingouin as pg 
import matplotlib.pyplot as plt 

from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity

def latex_mediation_table(
      med_DataFrame,
      caption="Mediation analysis",
      label="tab:mediation",
      p_sci_threshold_low=1e-3,
      p_sci_threshold_high=1e3,
      coef_sigfig=3,
      se_sigfig=3,
      ci_sigfig=3,
      x_name=None,   # NEW: name of X column to show in paths
      y_name=None,   # NEW: name of Y (target) to show in paths
      ):
    """
    Convert a pingouin.mediation_analysis(...) DataFrame into a LaTeX longtable.
    Columns expected: 'path', 'coef', 'se', 'pval', 'CI[2.5%]', 'CI[97.5%]', 'sig'
    Replaces 'X' and 'Y' in the 'path' column with x_name and y_name when provided.
    """
    import math, re

    def _fmt_sci(x, sig=3):
        s = f"{x:.{sig}e}"
        base, expo = s.split("e")
        expo = expo.replace("+", "")
        return f"${base}\\mathrm{{e}}{{{expo}}}$"

    def _fmt_num(x, sig=6, sci_if_small=True):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "NA"
        ax = abs(x)
        if (sci_if_small and (ax != 0 and ax < 1e-4)) or ax >= 1e4:
            return _fmt_sci(x, sig if sig <= 6 else 6)
        return f"{x:.{sig}g}"

    def _fmt_p(p):
        if p == 0.0:
            return r"$0.0\mathrm{e}{+00}$"
        if (p > 0) and (p < p_sci_threshold_low):
            return _fmt_sci(p, 3)
        if p >= p_sci_threshold_high:
            return _fmt_sci(p, 3)
        return f"{p:.3g}"

    def _latex_escape(s: str) -> str:
        # escape only problematic characters that may appear in column names
        return (s.replace("&", r"\&")
                 .replace("%", r"\%")
                 .replace("_", r"\_")
                 .replace("#", r"\#")
                 .replace("$", r"\$"))

    x_tex = _latex_escape(x_name) if x_name else None
    y_tex = _latex_escape(y_name) if y_name else None

    def _pretty_path(pth: str) -> str:
        p = str(pth)
        # Replace '... ~ X' with '... ~ <x_name>'
        if x_tex:
            p = re.sub(r'^(.*?)\s*~\s*X$', lambda m: f"{m.group(1)}\\\\ $\\sim$ {x_tex}", p)
        # Replace 'Y ~ ...' with '<y_name> ~ ...'
        if y_tex:
            p = re.sub(r'^Y\s*~\s*(.*)$', lambda m: f"{y_tex}\\\\ $\\sim$ {m.group(1)}", p)
        return p 

    rows = []
    for row in med_DataFrame.to_dict("records"):
        path = _pretty_path(row.get("path", ""))
        coef = _fmt_num(row.get("coef"), coef_sigfig)
        se = _fmt_num(row.get("se"), se_sigfig)
        p = _fmt_p(float(row.get("pval")))
        lo = _fmt_num(row.get("CI[2.5%]"), ci_sigfig)
        hi = _fmt_num(row.get("CI[97.5%]"), ci_sigfig)
        sig = str(row.get("sig", ""))
        rows.append(f"{path} & {coef} & {se} & {p} & {lo} & {hi} & {sig} \\\\")

    table = r"""\begin{longtable}{lrrrrrc}
            \caption{""" + caption + r"""}\label{""" + label + r"""}\\
            \toprule
            Path & Coef. & SE & $p$ & CI 2.5\% & CI 97.5\% & Sig \\
            \midrule
            \endfirsthead
            \toprule
            Path & Coef. & SE & $p$ & CI 2.5\% & CI 97.5\% & Sig \\
            \midrule
            \endhead
            \bottomrule
            \endfoot
            """ + "\n".join(rows) + "\n\\end{longtable}\n"
    return table

def zscore(s):
      v = s.std(ddof=0)  # population SD; avoids tiny-sample inflation
      return (s - s.mean()) / v if v and not np.isnan(v) else s*0

def plot_anova():
    return 

import matplotlib.pyplot as plt
import networkx as nx

import matplotlib.pyplot as plt
import networkx as nx

def plot_significant_mediations(meddfs, figsize=(8,6)):
    """
    Plot a single triangle diagram with all mediators whose indirect effect is significant.
    Criteria: both 'M ~ X' and 'Y ~ M' paths are sig == 'Yes'.
    """
    G = nx.DiGraph()
    x_node, y_node = "X", "Y"
    mediators = []

    # collect all mediators with significant indirect
    for df in meddfs:
        # find candidates
        sig = df[df['sig'] == 'Yes']
        if sig.empty:
            continue
        m = None
        has_xm, has_my = False, False
        xm_label, my_label = None, None

        for _, row in sig.iterrows():
            path = row['path']
            coef, pval = row['coef'], row['pval']
            if " ~ X" in path:   # Mediator ~ X
                m = path.split("~")[0].strip()
                has_xm = True
                xm_label = f"{coef:.3f}, p={pval:.3g}"
            if "Y ~" in path:   # Y ~ Mediator
                parts = path.split("~")
                if len(parts) == 2:
                    left, right = [s.strip() for s in parts]
                    if left == "Y":
                        m = right
                        has_my = True
                        my_label = f"{coef:.3f}, p={pval:.3g}"

        if has_xm and has_my and m is not None:
            mediators.append((m, xm_label, my_label))

    if not mediators:
        print("No significant indirect mediations found.")
        return

    # layout: put X at left, Y at right, mediators in between
    pos = {x_node: (0,0), y_node: (4,0)}
    for i, (m, _, _) in enumerate(mediators):
        pos[m] = (2, i*1.5)  # stagger mediators vertically

    # add edges
    for m, xm_label, my_label in mediators:
        G.add_edge(x_node, m, label=xm_label)
        G.add_edge(m, y_node, label=my_label)

    # draw
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=3000, font_size=12, font_weight="bold",
            arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.title("Significant Indirect Mediation Paths")
    plt.axis("off")
    plt.show()

def plot_mediation(DataFrame, Roles, Targets, Metrics, Mediator='Message Email Similarity', Seed=1, Alpha=0.05):
    mediations = []
    meddfs = []
    for Target in Targets:
        for Role in Roles:
            for Metric in Metrics: 
                if(Role == 'Student'):
                    df = DataFrame[DataFrame['Role'] == 'user']
                elif(Role == 'Teacher'):
                    df = DataFrame[DataFrame['Role'] == 'system']
                elif(Role == 'Student and Teacher'):
                    df = DataFrame
                else:
                    print(Role)
                    print("Not recognized")
                    assert(False)
                    
                cols = [Metric,
                        Mediator,
                        Target]
                
                # Ensure numeric (won’t touch if already numeric)
                #DataFrame[cols] = DataFrame[cols].apply(pd.to_numeric, errors='coerce')
                df.loc[:,cols] = df[cols].apply(pd.to_numeric, errors='coerce')

                # Z-score in place
                #DataFrame[cols] = DataFrame[cols].apply(zscore)
                df.loc[:,cols] = df[cols].apply(zscore)

                if(len(df) < 5): continue 
                # Mediation on standardized variables
                med = pg.mediation_analysis(
                data=df,
                x=Metric,
                m=Mediator,
                y=Target,
                alpha=Alpha,
                seed=Seed
                )
                #print(med)
                meddfs.append(med)
                mediations.append(latex_mediation_table(med, 
                                                        x_name=Metric, 
                                                        y_name=Target,
                                                        caption="Mediation analysis " + Metric + " on " + Target + " by " + " Messages " + Role ,
                                                        label="tab:mediation " + Metric + " on " + Target + " by " + " Messages " + Role))
    return meddfs, mediations

def plot_regression(DataFrame, Targets):
    # Plot regression using binned means
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax_idx, (ax, target) in enumerate(zip(axes.flat, Targets)):
        sub = DataFrame[["Message Email Similarity", 'Role', target]].copy()
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