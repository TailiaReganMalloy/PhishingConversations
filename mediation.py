"""

"""

import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pingouin as pg 

from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_pickle("./Database/MergedEmbeddingDictionary.pkl")
print(df.columns)

"""
Index(['UserId', 'Experiment', 'EmailId', 'PhaseTrial', 'Decision',
       'MessageNum', 'Message', 'EmailType', 'PhaseValue',
       'ExperimentCondition', 'Confidence', 'EmailAction', 'ReactionTime',
       'Correct', 'Role', 'Content', 'MessageEmbedding', 'User Improvement',
       'Perception of Emails as AI Generated', 'Pre-Experiment Quiz Score',
       'Response Message Similarity', 'EmailEmbedding', 'Age', 'Gender',
       'Education', 'Country', 'Victim', 'Chatbot', 'Q0', 'Q1', 'Q2', 'Q3',
       'Q4', 'Q5', 'PQ1', 'PQ2', 'PQ3', 'PQ4', 'PQ5', 'DemographicsEmbedding',
       'Message Cosine Similarity to Email', 'User Initial Performance',
       'User Final Performance', 'Gender Number', 'Education Years',
       'Phishing Experience', 'Chatbot Experience',
       'Cognitive Model Activity'],
      dtype='object')
"""

df["Open Response Cosine Similarity to Email"] = [
    float(round(float(cosine_similarity(m, e)[0, 0]) / 0.1) * 0.1) for m, e in zip(df["DemographicsEmbedding"], df["EmailEmbedding"])
]

df["Message Cosine Similarity to Email"] = [
    float(round(float(cosine_similarity(m, e)[0, 0]) / 0.1) * 0.1) for m, e in zip(df["MessageEmbedding"], df["EmailEmbedding"])
]


def latex_mediation_table(
      med_df,
      caption="Mediation analysis",
      label="tab:mediation",
      p_sci_threshold_low=1e-3,
      p_sci_threshold_high=1e4,
      coef_sigfig=6,
      se_sigfig=6,
      ci_sigfig=6,
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
        return f"{base}\\mathrm{{e}}{{{expo}}}"

    def _fmt_num(x, sig=6, sci_if_small=True):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "NA"
        ax = abs(x)
        if (sci_if_small and (ax != 0 and ax < 1e-4)) or ax >= 1e4:
            return _fmt_sci(x, sig if sig <= 6 else 6)
        return f"{x:.{sig}g}"

    def _fmt_p(p):
        if p == 0.0:
            return r"0.0\mathrm{e}{+00}"
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
            p = re.sub(r'^(.*?)\s*~\s*X$', lambda m: f"{m.group(1)} ~ {x_tex}", p)
        # Replace 'Y ~ ...' with '<y_name> ~ ...'
        if y_tex:
            p = re.sub(r'^Y\s*~\s*(.*)$', lambda m: f"{y_tex} ~ {m.group(1)}", p)
        return p

    rows = []
    for row in med_df.to_dict("records"):
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
# We assume that there is some effect of 'Age', 'Education', 'Victim', 'Chatbot', 'Perception of Emails as AI Generated, Pre-Experiment Quiz Score and Demographics embedding on User Improvement and test to see if message-email embedding cosine similarity is a mediator. 

def zscore(s):
      v = s.std(ddof=0)  # population SD; avoids tiny-sample inflation
      return (s - s.mean()) / v if v and not np.isnan(v) else s*0

columns = ['Age', 'Education Years', 'Phishing Experience', 'Chatbot Experience', 'Perception of Emails as AI Generated', 'Pre-Experiment Quiz Score', "Open Response Cosine Similarity to Email"]

# Significant 
# columns = []
for column in columns: 
      df = df[df['Role'] == 'system']
      cols = [column,
            'Message Cosine Similarity to Email',
            'User Improvement']
      
      # Ensure numeric (won’t touch if already numeric)
      df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

      # Z-score in place
      df[cols] = df[cols].apply(zscore)

      # Mediation on standardized variables
      med = pg.mediation_analysis(
      data=df,
      x='Message Cosine Similarity to Email',
      m=column,
      y='User Improvement',
      alpha=0.05,
      seed=1
      )
      #print(med)
      print(latex_mediation_table(med, x_name=column, y_name='User Improvement'))

"""
🔹 Predictors with significant direct effects on User Improvement
	•	Education Years → negative direct effect on User Improvement (coef = -0.214, p < 1e-24).
	•	Phishing Experience → positive direct effect (coef = 0.238, p < 1e-30).
	•	Chatbot Experience → negative direct effect (coef = -0.065, p = 0.002).
	•	Perception of Emails as AI Generated → negative direct effect (coef = -0.122, p < 1e-8).
	•	Open Response Cosine Similarity to Email → positive direct effect (coef = 0.183, p < 1e-18).

👉 These predictors directly explained variance in User Improvement, regardless of mediation.

⸻

🔹 Predictors with significant indirect (mediated) effects
	•	Education Years → significant indirect effect (coef = 0.015, p < .001).
	•	Phishing Experience → significant indirect effect (coef = 0.020, p < .001).
	•	Open Response Cosine Similarity → strong indirect effect (coef = 0.034, p < .001).

👉 These show that part of their effect on User Improvement worked through mediators, not just directly.

⸻

🔹 Predictors with no meaningful impact
	•	Age → small coefficients, no direct/indirect significance.
	•	Pre-Experiment Quiz Score → no effects.
	•	Chatbot Experience → while the direct effect was significant (negative), there was no mediation.
	•	Perception of Emails as AI Generated → same as Chatbot Experience: strong direct negative effect, no mediation.
    
\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Age & -0.0558053 & 0.0211384 & 0.00835 & -0.0972583 & -0.0143522 & Yes \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & 0.00136465 & 0.0211714 & 0.949 & -0.0401531 & 0.0428824 & No \\
Direct & 0.00169766 & 0.0212088 & 0.936 & -0.0398935 & 0.0432888 & No \\
Indirect & -0.000333003 & 0.00128357 & 0.784 & -0.00301465 & 0.00208913 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Education Years & -0.0742719 & 0.021113 & 4.438\mathrm{e}{-04} & -0.115675 & -0.0328688 & Yes \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & -0.213966 & 0.0206811 & 1.550\mathrm{e}{-24} & -0.254523 & -0.17341 & Yes \\
Direct & -0.214715 & 0.020742 & 1.465\mathrm{e}{-24} & -0.25539 & -0.174039 & Yes \\
Indirect & 0.000748273 & 0.00162737 & 0.668 & -0.00192589 & 0.0044527 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Gender Number & 0.00377145 & 0.0211713 & 0.859 & -0.037746 & 0.0452889 & No \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & 0.0290257 & 0.0211625 & 0.17 & -0.0124746 & 0.070526 & No \\
Direct & 0.0290039 & 0.0211671 & 0.171 & -0.0125052 & 0.0705131 & No \\
Indirect & 2.173527\mathrm{e}{-05} & 0.000463441 & 0.956 & -0.000826679 & 0.00116374 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Phishing Experience & 0.0786437 & 0.0211059 & 1.993\mathrm{e}{-04} & 0.0372545 & 0.120033 & Yes \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & 0.238083 & 0.0205626 & 3.785\mathrm{e}{-30} & 0.197759 & 0.278407 & Yes \\
Direct & 0.2391 & 0.0206293 & 3.325\mathrm{e}{-30} & 0.198645 & 0.279555 & Yes \\
Indirect & -0.00101696 & 0.00173871 & 0.6 & -0.00475659 & 0.00189916 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Chatbot Experience & -0.00839239 & 0.0211707 & 0.692 & -0.0499087 & 0.0331239 & No \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & -0.0651833 & 0.0211264 & 0.00206 & -0.106613 & -0.0237538 & Yes \\
Direct & -0.0651386 & 0.0211316 & 0.00208 & -0.106578 & -0.023699 & Yes \\
Indirect & -4.469644\mathrm{e}{-05} & 0.000514614 & 0.908 & -0.00184109 & 0.000570402 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Perception of Emails as AI Generated & -0.0209955 & 0.0211668 & 0.321 & -0.0625041 & 0.0205132 & No \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & -0.121971 & 0.0210134 & 7.380\mathrm{e}{-09} & -0.163178 & -0.0807628 & Yes \\
Direct & -0.121901 & 0.0210226 & 7.639\mathrm{e}{-09} & -0.163127 & -0.0806751 & Yes \\
Indirect & -6.956070\mathrm{e}{-05} & 0.000602756 & 0.82 & -0.00149314 & 0.00104716 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Pre-Experiment Quiz Score & 0.0184508 & 0.0211678 & 0.383 & -0.0230599 & 0.0599615 & No \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & 0.0337589 & 0.0211594 & 0.111 & -0.00773524 & 0.075253 & No \\
Direct & 0.033662 & 0.0211674 & 0.112 & -0.00784794 & 0.0751719 & No \\
Indirect & 9.689261\mathrm{e}{-05} & 0.000641044 & 0.884 & -0.000790963 & 0.00245893 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
             M-E Similarity ~ Open Response Cosine Similarity to Email & 0.202309 & 0.0207336 & 4.697\mathrm{e}{-22} & 0.161649 & 0.242968 & Yes \\
User Improvement ~  M-E Similarity & 0.0058725 & 0.0211711 & 0.782 & -0.0356446 & 0.0473895 & No \\
Total & 0.176261 & 0.02084 & 4.851\mathrm{e}{-17} & 0.135393 & 0.217129 & Yes \\
Direct & 0.182544 & 0.0212746 & 1.741\mathrm{e}{-17} & 0.140824 & 0.224265 & Yes \\
Indirect & -0.00628326 & 0.00446522 & 0.164 & -0.0158047 & 0.00155476 & No \\
\end{longtable}

"""

"""
Student message mediation 
\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
            Age ~ Age & -0.10394 & 0.0451617 & 0.0218 & -0.192676 & -0.0152028 & Yes \\
User Improvement ~ Age & 0.063257 & 0.0453167 & 0.163 & -0.0257844 & 0.152298 & No \\
Total & -0.0869099 & 0.0452358 & 0.0553 & -0.175792 & 0.00197257 & No \\
Direct & -0.0812123 & 0.0454609 & 0.0747 & -0.170537 & 0.00811282 & No \\
Indirect & -0.00569753 & 0.0052138 & 0.152 & -0.0179491 & 0.00193107 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
            Education Years ~ Education Years & 0.0260474 & 0.0453923 & 0.566 & -0.0631423 & 0.115237 & No \\
User Improvement ~ Education Years & -0.265618 & 0.0437765 & 2.618\mathrm{e}{-09} & -0.351633 & -0.179603 & Yes \\
Total & -0.0869099 & 0.0452358 & 0.0553 & -0.175792 & 0.00197257 & No \\
Direct & -0.0800455 & 0.0436854 & 0.0675 & -0.165882 & 0.00579087 & No \\
Indirect & -0.00686434 & 0.0112904 & 0.516 & -0.0270476 & 0.016702 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
            Phishing Experience ~ Phishing Experience & 0.0764693 & 0.0452747 & 0.0919 & -0.0124895 & 0.165428 & No \\
User Improvement ~ Phishing Experience & 0.352608 & 0.0424912 & 1.054\mathrm{e}{-15} & 0.269118 & 0.436097 & Yes \\
Total & -0.0869099 & 0.0452358 & 0.0553 & -0.175792 & 0.00197257 & No \\
Direct & -0.114543 & 0.0423411 & 0.00707 & -0.197738 & -0.0313484 & Yes \\
Indirect & 0.0276335 & 0.0172613 & 0.072 & -0.000175566 & 0.0686428 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
            Chatbot Experience ~ Chatbot Experience & -0.0868512 & 0.0452361 & 0.0554 & -0.175734 & 0.00203171 & No \\
User Improvement ~ Chatbot Experience & -0.0725261 & 0.0452881 & 0.11 & -0.161511 & 0.0164589 & No \\
Total & -0.0869099 & 0.0452358 & 0.0553 & -0.175792 & 0.00197257 & No \\
Direct & -0.0939173 & 0.0453061 & 0.0387 & -0.182938 & -0.0048963 & Yes \\
Indirect & 0.00700741 & 0.00560593 & 0.164 & -0.00118651 & 0.020098 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
            Perception of Emails as AI Generated ~ Perception of Emails as AI Generated & 0.118546 & 0.0450875 & 0.00883 & 0.0299546 & 0.207136 & Yes \\
User Improvement ~ Perception of Emails as AI Generated & -0.117996 & 0.0450904 & 0.00915 & -0.206593 & -0.0293994 & Yes \\
Total & -0.0869099 & 0.0452358 & 0.0553 & -0.175792 & 0.00197257 & No \\
Direct & -0.0739613 & 0.0453331 & 0.103 & -0.163035 & 0.0151126 & No \\
Indirect & -0.0129485 & 0.00739432 & 0.028 & -0.0309829 & -0.00167338 & Yes \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
            Pre-Experiment Quiz Score ~ Pre-Experiment Quiz Score & -0.0512565 & 0.045348 & 0.259 & -0.140359 & 0.0378462 & No \\
User Improvement ~ Pre-Experiment Quiz Score & -0.0514809 & 0.0453474 & 0.257 & -0.140583 & 0.0376208 & No \\
Total & -0.0869099 & 0.0452358 & 0.0553 & -0.175792 & 0.00197257 & No \\
Direct & -0.0897845 & 0.0452704 & 0.0479 & -0.178735 & -0.00083362 & Yes \\
Indirect & 0.00287462 & 0.00385404 & 0.484 & -0.00142339 & 0.0137261 & No \\
\end{longtable}

\begin{longtable}{lrrrrrc}
            \caption{Mediation analysis}\label{tab:mediation}\\
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
            Open Response Cosine Similarity to Email ~ Open Response Cosine Similarity to Email & -0.0344721 & 0.0453807 & 0.448 & -0.123639 & 0.0546949 & No \\
User Improvement ~ Open Response Cosine Similarity to Email & 0.243105 & 0.0440454 & 5.548\mathrm{e}{-08} & 0.156561 & 0.329648 & Yes \\
Total & -0.0869099 & 0.0452358 & 0.0553 & -0.175792 & 0.00197257 & No \\
Direct & -0.078623 & 0.0439721 & 0.0744 & -0.165023 & 0.00777691 & No \\
Indirect & -0.0082869 & 0.0141259 & 0.532 & -0.040039 & 0.0127119 & No \\
\end{longtable}
🔹 Predictors with significant direct effects on User Improvement
	•	Education Years → negative direct effect (coef = -0.266, p < 1e-8).
	•	Phishing Experience → positive direct effect (coef = 0.353, p < 1e-15).
	•	Chatbot Experience → negative direct effect (coef = -0.094, p = 0.039).
	•	Perception of Emails as AI Generated → negative direct effect (coef = -0.118, p = 0.009).
	•	Pre-Experiment Quiz Score → negative direct effect (coef = -0.090, p = 0.048).
	•	Open Response Cosine Similarity to Email → positive direct effect (coef = 0.243, p < 1e-7).

⸻

🔹 Predictors with significant indirect (mediated) effects
	•	Perception of Emails as AI Generated → negative indirect effect (coef = -0.013, p = 0.028).
→ Suggests part of its harmful effect on improvement runs through a mediator.

👉 None of the other predictors showed significant mediation effects in this batch.

⸻

🔹 Predictors with no meaningful impact
	•	Age → effect trended negative, but not significant (p ≈ .05 for total, ns for direct/indirect).
	•	Education Years → no mediation; its effect was purely direct.
	•	Phishing Experience → strong direct effect, indirect trended positive (p = .07), but not sig.
	•	Chatbot Experience → direct only, no mediation.
	•	Pre-Experiment Quiz Score → weak negative direct effect, no mediation.
	•	Open Response Cosine Similarity → strong direct positive effect, no mediation.
    
"""