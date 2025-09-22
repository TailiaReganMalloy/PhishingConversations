import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import pandas as pd 
import seaborn as sns 
import pingouin as pg 
import matplotlib.pyplot as plt 

import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

import tqdm 
import json  
import sklearn 
import scipy 
import re
from collections import Counter

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


        # Bin similarity by 0.001 (round to 2 decimals) and aggregate per bin+role
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

# ------------------------
# Prepare per-message features for columns 2-4
# ------------------------
def build_message_features(messages_df, emails_df):
    rows = []
    for _, msg in messages_df.iterrows():
        msg_text = msg.get('Message', "") or ""
        # find email row
        col = emails_df.loc[emails_df['EmailId'] == msg.get('Email Id')]
        if col.empty:
            email_text = ""
        else:
            r = col.iloc[0]
            email_text = f"Sender: {r['Sender']}\nSubject: {r['Subject']}\n{r['Body']}"

        msg_tokens = tokenize(msg_text)
        email_tokens = tokenize(email_text)

        msg_set = set(msg_tokens)
        email_set = set(email_tokens)
        prop_common = (len(msg_set & email_set) / len(msg_set)) if len(msg_set) > 0 else 0.0
        
        bigr_msg = make_ngrams(msg_tokens, 2)
        bigr_email = make_ngrams(email_tokens, 2)
        tri_msg = make_ngrams(msg_tokens, 3)
        tri_email = make_ngrams(email_tokens, 3)
        qua_msg =  make_ngrams(msg_tokens, 4)
        qua_email = make_ngrams(msg_tokens, 4)

        # count shared ngram occurrences (use set intersection counts to avoid duplicates)
        shared_bigrams = len(set(bigr_msg) & set(bigr_email))
        shared_trigrams = len(set(tri_msg) & set(tri_email))
        shared_quadgrams = len(set(qua_msg) & set(qua_email))
        ngram_counts = shared_bigrams + shared_trigrams + shared_quadgrams

        # correct categorization (ensure numeric)
        correct = msg.get('Correct Categorization', np.nan)
        try:
            correct = float(correct)
        except Exception:
            correct = np.nan

        msg_length = np.clip(len(msg_tokens) / len(email_tokens), 0.1, 1)

        rows.append({
            "Message Id": msg.get('Message Id'),
            "msg_length": msg_length,
            "prop_common": prop_common,
            "ngram_counts": ngram_counts,
            "correct": correct
        })
    return pd.DataFrame(rows)

def tokenize(text):
    return re.findall(r"\b\w+\b", (text or "").lower())

def make_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

# composite: mean of normalized parts (skip NaNs)
def composite_row(r):
    parts = []
    for c in ('mean_msg_length_norm','mean_prop_common_norm','mean_ngram_counts_norm'):
        v = r.get(c, np.nan)
        if pd.notna(v):
            parts.append(v)
    return np.nan if len(parts) == 0 else float(np.mean(parts))
    
def compare_embeddings(emails, orig):
    orig = orig.copy()
    orig['Message Id'] = orig.index

    # make a unique mapping on Message (keep first if duplicates)
    orig_unique = orig.drop_duplicates(subset='Message', keep='first')[['Message', 'Message Id']]

    # load merged and merge in MessageId
    merged_path = "./Database/Merged.pkl"
    merged = pd.read_pickle(merged_path)

    merged = merged.merge(orig_unique, on='Message', how='left')

    # save to a new file (or overwrite merged_path if you prefer)
    merged.to_pickle("./Database/Merged_withMessageId.pkl")
    print("Saved ./Database/Merged_withMessageId.pkl")

    sources = {
        "BGE": "./Database/AdditionalEmbeddings/BGE/OpenEmailEmbeddings.pkl",
        "Gemma": "./Database/AdditionalEmbeddings/Gemma/OpenEmailEmbeddings.pkl",
        "Granite": "./Database/AdditionalEmbeddings/Granite/OpenEmailEmbeddings.pkl",
        "Qwen3-0.6B": "./Database/AdditionalEmbeddings/Qwen3-0.6B/OpenEmailEmbeddings.pkl",
        "Qwen3-4B": "./Database/AdditionalEmbeddings/Qwen3-4B/OpenEmailEmbeddings.pkl",
        "Qwen3-8B": "./Database/AdditionalEmbeddings/Qwen3-8B/OpenEmailEmbeddings.pkl",
        "OpenAI": "./Database/AdditionalEmbeddings/OpenAI/EmailEmbeddings.pkl",
        "OpenAI Ada": "./Database/AdditionalEmbeddings/OpenAI-Ada/EmailEmbeddings.pkl",
        "MiniLM-L6": "./Database/AdditionalEmbeddings/MiniLM-L6/OpenEmailEmbeddings.pkl"
    }

    dfs = []
    for name, path in sources.items():
        df = pd.read_pickle(path)
        df["source"] = name            # optional: keep track of origin
        dfs.append(df)

    email_embeddings = pd.concat(dfs, ignore_index=True)

    sources = {
        "BGE": "./Database/AdditionalEmbeddings/BGE/OpenMessageEmbeddings.pkl",
        "Gemma": "./Database/AdditionalEmbeddings/Gemma/OpenMessageEmbeddings.pkl",
        "Granite": "./Database/AdditionalEmbeddings/Granite/OpenMessageEmbeddings.pkl",
        "Qwen3-0.6B": "./Database/AdditionalEmbeddings/Qwen3-0.6B/OpenMessageEmbeddings.pkl",
        "Qwen3-4B": "./Database/AdditionalEmbeddings/Qwen3-4B/OpenMessageEmbeddings.pkl",
        "Qwen3-8B": "./Database/AdditionalEmbeddings/Qwen3-8B/OpenMessageEmbeddings.pkl",
        "OpenAI": "./Database/AdditionalEmbeddings/OpenAI/MessageEmbeddings.pkl",
        "OpenAI Ada": "./Database/AdditionalEmbeddings/OpenAI-Ada/MessageEmbeddings.pkl",
        "MiniLM-L6": "./Database/AdditionalEmbeddings/MiniLM-L6/OpenMessageEmbeddings.pkl"
    }

    dfs = []
    for name, path in sources.items():
        df = pd.read_pickle(path)
        df["source"] = name            # optional: keep track of origin
        dfs.append(df)

    message_embeddings = pd.concat(dfs, ignore_index=True)

    messages = pd.read_pickle("./Database/Merged_withMessageId.pkl")

    """
    Index(['User Id', 'Experiment', 'Email Id', 'Phase Trial', 'Decision',
        'Message Num', 'Message', 'Email Type', 'Phase Value',
        'Experiment Condition', 'Categorization Confidence', 'Email Action',
        'Reaction Time', 'Correct Categorization', 'Role', 'Content',
        'Message Embedding', 'User Improvement', 'AI Generation Perception',
        'Pre Experiment Quiz Score', 'Response Message Similarity',
        'Email Embedding', 'Age', 'Gender', 'Education', 'Country', 'Victim',
        'Chatbot', 'Question 0', 'Question 1', 'Question 2', 'Question 3',
        'Question 4', 'Question 5', 'Post Experiment Question 1',
        'Post Experiment Question 2', 'Post Experiment Question 3',
        'Post Experiment Question 4', 'Open Response',
        'Open Response Embedding', 'Message Email Similarity',
        'User Initial Performance', 'User Final Performance', 'Gender Number',
        'Education Years', 'Phishing Experience', 'Chatbot Experience',
        'Cognitive Model Activity'],
        dtype='object')
    """
    models = message_embeddings["Model"].unique()

    current_models = ['text-embedding-3-large', './Models/bge-large-en-v1.5' ,'./Models/embeddinggemma-300m',
    './Models/granite-embedding-small-english-r2',
    './Models/qwen3-embedding-0.6B', './Models/Qwen3-Embedding-4B',
    './Models/Qwen3-Embedding-8B' ,'text-embedding-ada-002',
    'text-embedding-3-small' , './Models/MiniLM-L6-v2']

    to_replace = ['OpenAI\nLarge 3', 'BAAI BGE-Large', 'Google Gemma',
    'IBM Granite', 'Qwen3 0.6B' ,'Qwen3 4B',
    'Qwen3 8B', 'OpenAI ADA',
    'OpenAI\nSmall 3',  'MiniLM-L6']

    # Build a mapping from old display names -> desired model keys and apply to both dataframes
    mapping = dict(zip(current_models, to_replace))

    # Replace model column values (no-op for values not in mapping)
    message_embeddings['Model'] = message_embeddings['Model'].replace(mapping)
    email_embeddings['Model'] = email_embeddings['Model'].replace(mapping)

    # recompute models after replacement
    #message_embeddings = message_embeddings[message_embeddings["Model"] != 'Google Gemma']
    models = to_replace #message_embeddings["Model"].unique()

    # --- replace existing rows collection + single-panel plotting with a 4-column plot ---

    rows = []
    total = len(models) * len(messages)

    with tqdm.tqdm(total = total) as pbar:
        for model in models:
            model_message_embeddings = message_embeddings[message_embeddings["Model"] == model]
            model_email_embeddings = email_embeddings[email_embeddings["Model"] == model]

            similarity = []
            correct = []
            confidence = []
            reaction = []
            msg_length = []
            prop_common_words = []
            shared_bigrams = []
            shared_trigrams = []
            shared_quadgrams = []

            for midx, message in messages.iterrows():
                message_embedding = model_message_embeddings[model_message_embeddings['MessageId'] == message['Message Id']]
                email_embedding = model_email_embeddings[model_email_embeddings['EmailId'] == message['Email Id']]

                if len(message_embedding) == 0 or len(email_embedding) == 0:
                    pbar.update(1)
                    continue

                # load embeddings
                try:
                    message_emb = np.array(json.loads(message_embedding['Embedding'].item())).reshape(1, -1)
                    email_emb = np.array(json.loads(email_embedding['Embedding'].item())).reshape(1, -1)
                except Exception:
                    pbar.update(1)
                    continue

                # cosine similarity
                sim = float(sklearn.metrics.pairwise.cosine_similarity(message_emb, email_emb)[0][0])
                similarity.append(sim)

                # categorization accuracy (expect 0/1 or similar)
                correct.append(float(message.get('Correct Categorization', np.nan)))
                confidence.append(float(message.get('Categorization Confidence', np.nan)))
                reaction.append(float(message.get('Reaction Time', np.nan)))

                # build plain text for comparison (first matching email row)
                col = emails.loc[emails['EmailId'] == message['Email Id']]
                if col.empty:
                    email_text = ""
                else:
                    row = col.iloc[0]
                    email_text = f"Sender: {row['Sender']}\nSubject: {row['Subject']}\n{row['Body']}"

                # tokens and lengths
                msg_tokens = tokenize(message['Message'])
                email_tokens = tokenize(email_text)
                msg_length.append(np.clip(len(msg_tokens) / len(email_tokens), 0.2, 1))

                # proportion of distinct message words that appear in email
                msg_set = set(msg_tokens)
                email_set = set(email_tokens)
                prop_common = (len(msg_set & email_set) / len(msg_set)) if len(msg_set) > 0 else 0.0
                prop_common_words.append(prop_common)

                # shared n-grams (unique)
                msg_bigrams = set(make_ngrams(msg_tokens, 2))
                email_bigrams = set(make_ngrams(email_tokens, 2))
                shared_bigrams.append(len(msg_bigrams & email_bigrams))
                msg_trigrams = set(make_ngrams(msg_tokens, 3))
                email_trigrams = set(make_ngrams(email_tokens, 3))
                shared_trigrams.append(len(msg_trigrams & email_trigrams))
                msg_trigrams = set(make_ngrams(msg_tokens, 4))
                email_trigrams = set(make_ngrams(email_tokens, 4))
                shared_quadgrams.append(len(msg_trigrams & email_trigrams))

                pbar.update(1)
            
            ngram_counts = [b + t + q for b, t, q in zip(shared_bigrams, shared_trigrams, shared_quadgrams)] 

            arrs = {
                "Similarity": np.asarray(similarity),
                "Correct": np.asarray(correct),
                "Confidence": np.asarray(confidence),
                "Reaction": np.asarray(reaction),
                "Proportion": np.asarray(prop_common_words),
                "N-Gram Count": np.asarray(ngram_counts),
                "Length": np.asarray(msg_length)
            }

            # align lengths (truncate to shortest) to avoid length mismatch errors
            lengths = [a.shape[0] for a in arrs.values() if a is not None]
            if len(set(lengths)) > 1:
                minlen = min(lengths)
                for k in list(arrs.keys()):
                    arrs[k] = arrs[k][:minlen]

            # build dataframe
            sdf = pd.DataFrame(arrs)

            # --- Normalize selected columns to range [0,1] to ensure comparability ---
            def normalize_series(s):
                s = pd.to_numeric(s, errors='coerce')
                if s.isna().all():
                    return s
                minv = s.min()
                maxv = s.max()
                if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
                    return s - minv if not pd.isna(minv) else s
                return (s - minv) / (maxv - minv)

            for col in ["Similarity", "Correct", "Confidence", "Reaction"]:
                if col in sdf.columns:
                    sdf[col] = normalize_series(sdf[col])

            # Bin similarity to nearest 0.001 (shows values rounded to 2 decimals)
            sdf["Similarity"] = (sdf["Similarity"] / 0.001).round().mul(0.001)
            sdf["Correct"] = (sdf["Correct"] / 0.001).round().mul(0.001)
            sdf["Confidence"] = (sdf["Confidence"] / 0.001).round().mul(0.001)
            sdf["Reaction"] = (sdf["Reaction"] / 0.001).round().mul(0.001)

            if len(similarity) >= 2:
                grp = sdf.groupby(["Similarity"], as_index=False).mean()
                res_correct = linregress(grp["Similarity"], grp['Correct'])  
                r_correct = res_correct.rvalue 
                p_correct = res_correct.pvalue

                res_confidence = linregress(grp["Similarity"], grp['Confidence'])  
                r_confidence = res_confidence.rvalue 
                p_confidence = res_confidence.pvalue

                res_reaction = linregress(grp["Similarity"], grp['Reaction'])  
                r_reaction = res_reaction.rvalue
                p_reaction = res_reaction.pvalue

                # embedding size (try to read length of embedding vector)
                try:
                    Embedding_Size = len(json.loads(model_message_embeddings.iloc[0]['Embedding']))
                except Exception:
                    Embedding_Size = np.nan

                rows.append({
                    "model": model,
                    "r_correct": float(r_correct),
                    "p_correct": float(p_correct),
                    "r_confidence": float(r_confidence),
                    "p_confidence": float(p_confidence),
                    "r_reaction": float(r_reaction),
                    "p_reaction": float(p_reaction),
                    "Embedding Size": Embedding_Size
                })
                print(rows[-1])

    # build dataframe
    size_accuracy = pd.DataFrame(rows)
    #size_accuracy['Log Embedding Size'] = size_accuracy['Embedding Size'].apply(lambda s: np.log10(s) if pd.notna(s) and s > 0 else np.nan)

    msg_features = build_message_features(messages, emails)
    # drop rows without correct categorization
    msg_features = msg_features.dropna(subset=['correct'])

    # compute per-model mean message-level features (if present) and add a composite metric
    # if the per-model lists were recorded you'll have columns like mean_msg_length, mean_prop_common, mean_ngram_counts
    # otherwise compute approximate per-model means by scanning messages (fallback)
    if not {'mean_msg_length','mean_prop_common','mean_ngram_counts'}.issubset(size_accuracy.columns):
        # fallback: compute per-model means from msg_features (global per-message table built earlier)
        if 'msg_features' in globals() and not msg_features.empty:
            model_means = {}
            for model in size_accuracy['model'].unique():
                # try to identify messages used for that model by using message_embeddings presence
                model_rows = message_embeddings[message_embeddings['Model'] == model]
                mids = set(model_rows['MessageId'].dropna().astype(int).tolist()) if 'MessageId' in model_rows.columns else set()
                mf = msg_features[msg_features['Message Id'].isin(mids)] if mids else msg_features.copy()
                model_means[model] = {
                    'mean_msg_length': mf['msg_length'].mean() if not mf.empty else np.nan,
                    'mean_prop_common': mf['prop_common'].mean() if not mf.empty else np.nan,
                    'mean_ngram_counts': mf['ngram_counts'].mean() if not mf.empty else np.nan
                }
            # map into dataframe
            size_accuracy['mean_msg_length'] = size_accuracy['model'].map(lambda m: model_means.get(m, {}).get('mean_msg_length', np.nan))
            size_accuracy['mean_prop_common'] = size_accuracy['model'].map(lambda m: model_means.get(m, {}).get('mean_prop_common', np.nan))
            size_accuracy['mean_ngram_counts'] = size_accuracy['model'].map(lambda m: model_means.get(m, {}).get('mean_ngram_counts', np.nan))

    # normalize each metric across models to 0-1 for composite
    for col in ('mean_msg_length','mean_prop_common','mean_ngram_counts'):
        if col in size_accuracy.columns:
            col_min = size_accuracy[col].min(skipna=True)
            col_max = size_accuracy[col].max(skipna=True)
            if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
                size_accuracy[col + '_norm'] = (size_accuracy[col] - col_min) / (col_max - col_min)
            else:
                size_accuracy[col + '_norm'] = np.nan
        else:
            size_accuracy[col + '_norm'] = np.nan

    size_accuracy['composite_metric'] = size_accuracy.apply(composite_row, axis=1)
    # Top panel: per-model r_cat vs Log Embedding Size with composite shown as marker size/color
    comparisons = size_accuracy
    
    metric = 'r_mean'
    pmetric = 'p_correct'

    # overlay composite as sized markers (if available) - mean only (no std)
    comp_df = comparisons.dropna(subset=['composite_metric', metric, 'Embedding Size']).copy()
    return (comparisons, msg_features, comp_df)


# Helper to plot binned means on bottom panels (as before)
def plot_msg_vs_correct_binned(msg_features, ax, xcol, xlabel):
    df = msg_features.dropna(subset=[xcol, 'correct']).copy()
    if df.empty:
        ax.set_title(xlabel + " (no data)")
        return

    if xcol == 'prop_common':
        bins = np.linspace(0.0, 1.0, 21)
    else:
        uniq = df[xcol].nunique()
        bins = uniq if uniq <= 10 else 20

    sns.regplot(data=df, x=xcol, y='correct', ci=95, scatter=False, ax=ax, line_kws={"color":"C0", "alpha":0.9})

    binned = df.copy()
    binned['bin'] = pd.cut(binned[xcol], bins=bins, include_lowest=True)
    grouped = binned.groupby('bin').agg(
        mean_x=(xcol, 'mean'),
        mean_y=('correct', 'mean'),
        count=('correct', 'count'),
        std_y=('correct', 'std')
    ).reset_index(drop=True)
    grouped['sem'] = grouped.apply(lambda r: (r['std_y'] / np.sqrt(r['count'])) if r['count'] > 1 and not np.isnan(r['std_y']) else 0.0, axis=1)
    grouped = grouped.dropna(subset=['mean_x', 'mean_y'])

    ax.scatter(grouped['mean_x'], grouped['mean_y'], color='C1', alpha=0.75, zorder=7)

    #r, p = scipy.stats.pearsonr(df[xcol], df['correct'])
    res = linregress(grouped['mean_x'], grouped['mean_y']) 
    r2 = res.rvalue ** 2
    p = res.pvalue
    
    ax.text(0.05, 0.95, f"$r^2$={r2:.3f}, p={p:.2g}", transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8), fontsize=12)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Correct Categorization")
    x_min, x_max = df[xcol].min(), df[xcol].max()
    x_pad = max(0.001, 0.05 * (x_max - x_min if x_max > x_min else 1.0))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    y_min, y_max = df['correct'].min(), df['correct'].max()
    y_pad = max(0.001, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

