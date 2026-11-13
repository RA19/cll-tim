# -*- coding: utf-8 -*-
"""
CLL-TIM Showcase Figure Generator
---------------------------------
NEW analysis script - the original CLL-TIM application code is NOT touched.

Replicates the exact per-base-learner inference path of CLLTIM_MobileVer2.py
(feature-subset selection, NaN->column-mean imputation for non-XGBoost
learners, stored StandardScaler transforms, predict_proba, SHAP attribution)
and regenerates the showcase figures in docs/figures/.

Run from the repository root inside the pinned environment:

    conda run -n clltim_calc_env --no-capture-output ^
        python examples/clltim_showcase.py

All paths are relative to the repository root - no local absolute paths.
"""

import os
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import shap
import sys

# ------------------------------------------------------------------
# Paths (relative to repo root)
# ------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "CLLTIMData")
GOLD = os.path.join(REPO, "docs", "synthetic_patient_examples")
FIGDIR = os.path.join(REPO, "docs", "figures")
os.makedirs(FIGDIR, exist_ok=True)

rng = np.random.RandomState(42)

# Optional: python examples/clltim_showcase.py anatomy,votes,shap,missingness,pipeline
ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
def want(name):
    return ONLY is None or name in ONLY

# ------------------------------------------------------------------
# Load the same artifacts the application loads
# ------------------------------------------------------------------
Collect_CVData = np.load(os.path.join(D, "_Collect_CVData.npy"))
Collect_ModelNames = np.load(os.path.join(D, "_Collect_ModelNames.npy"))
ToKeep = np.load(os.path.join(D, "_28_ToKeep.npy"))
Collect_CVData_Chosen = Collect_CVData[ToKeep, :]
Collect_ModelNames_Chosen = Collect_ModelNames[ToKeep, :]
Collect_FeatIndies = np.load(
    os.path.join(D, "CLL_TIM_FeatIndexes.npy"), allow_pickle=True
)
N_Chosen = len(ToKeep)
ALLFeats = np.load(os.path.join(D, "ALLFeats_.npy"))
TransList = np.load(os.path.join(D, "TrainsListing.npy"), allow_pickle=True)

with open(os.path.join(D, "vecgetcolmean.pkl"), "rb") as f:
    vecgetcolmeanALL = pickle.load(f)
with open(os.path.join(D, "vec_kmeans.pkl"), "rb") as f:
    X_Train_summaries = pickle.load(f)
with open(os.path.join(D, "vec_scaler.pkl"), "rb") as f:
    scalerfits = pickle.load(f)
with open(os.path.join(D, "vec_mx_all.pkl"), "rb") as f:
    vec_mx = pickle.load(f)
with open(os.path.join(D, "vec_mn_all.pkl"), "rb") as f:
    vec_mn = pickle.load(f)

CurrBL_vec = []
for nn in range(N_Chosen):
    with open(os.path.join(D, "CLL-TIM_BLN_" + str(nn) + ".sav"), "rb") as f:
        CurrBL_vec.append(pickle.load(f))

ALGO_NAMES = [str(Collect_ModelNames_Chosen[nn, 1]) for nn in range(N_Chosen)]

# Union of the feature subsets used by the 28 base learners (228 features)
FAllInds = np.array([], dtype=int)
for nn in range(N_Chosen):
    FAllInds = np.append(
        FAllInds, Collect_FeatIndies.item()[Collect_CVData_Chosen[nn, 0]]
    )
FAInds = np.unique(FAllInds)
N_FEATS = len(FAInds)

# For each learner: positions of its raw feature indices inside the
# 228-dimensional patient vector (same construction as the app).
FEATS_OF_BL = []
for nn in range(N_Chosen):
    FInds = Collect_FeatIndies.item()[Collect_CVData_Chosen[nn, 0]]
    pos = np.array(
        [int(np.squeeze(np.argwhere(FAInds == ff))) for ff in FInds]
    )
    FEATS_OF_BL.append((FInds, pos))


def Nan2Mean_useColmean(Mat, get_colmean):
    """Identical logic to the helper in CLLTIM_MobileVer2.py."""
    Mat = Mat.copy()
    for j in range(Mat.shape[1]):
        col = Mat[:, j]
        miss = np.isnan(col)
        if np.any(miss):
            col[miss] = get_colmean[j]
            Mat[:, j] = col
    return Mat


def predict_all(OutputVal):
    """Run the 28-learner ensemble on a 228-dim patient feature vector.

    Returns (AllProbs (28,), FinalFeats (228,), FinalFeatsMat (28,228)).
    Mirrors CLLTIM_MobileVer2.py lines ~795-858 exactly.
    """
    AllProbs = np.zeros((N_Chosen, 1))
    FinalFeats = np.zeros((1, N_FEATS))
    FinalFeatsMat = np.zeros((N_Chosen, N_FEATS)) - 999

    for nn in range(N_Chosen):
        Curr_BL = CurrBL_vec[nn]
        FInds, pos = FEATS_OF_BL[nn]
        VMatrix = np.row_stack((OutputVal, OutputVal))
        VMatrix = VMatrix[:, pos]

        if Collect_ModelNames_Chosen[nn, 1] != "XGB":
            get_colmean = vecgetcolmeanALL[nn]
            VMatrix = Nan2Mean_useColmean(VMatrix, get_colmean)
            VMatrix = scalerfits[nn].transform(VMatrix)

        Probs = Curr_BL.predict_proba(VMatrix)
        AllProbs[nn] = Probs[0, 1]

    return AllProbs


def shap_all(OutputVal):
    """Per-learner SHAP values mapped back onto the 228-feature axis.

    Mirrors the app's aggregation: FinalFeatsMat[nn, cind] holds learner
    nn's SHAP value for global feature cind; FinalFeats sums them.
    """
    FinalFeats = np.zeros((1, N_FEATS))
    FinalFeatsMat = np.zeros((N_Chosen, N_FEATS)) * np.nan

    for nn in range(N_Chosen):
        Curr_BL = CurrBL_vec[nn]
        FInds, pos = FEATS_OF_BL[nn]
        VMatrix = np.row_stack((OutputVal, OutputVal))
        VMatrix = VMatrix[:, pos]

        name = str(Collect_ModelNames_Chosen[nn, 1])
        if name != "XGB":
            VMatrix = Nan2Mean_useColmean(VMatrix, vecgetcolmeanALL[nn])
            VMatrix = scalerfits[nn].transform(VMatrix)

        if name in ("Elastic", "LogisticRegression"):
            explainer = shap.KernelExplainer(
                Curr_BL.predict_proba, X_Train_summaries[nn]
            )
            shap_values = explainer.shap_values(VMatrix)
            sv = np.asarray(shap_values[0])
            sv = sv[0] if sv.ndim == 2 else np.asarray(sv).reshape(-1)
        else:
            explainer = shap.TreeExplainer(Curr_BL)
            sv = np.asarray(explainer.shap_values(VMatrix))
            # shape handling: (2,k,2) -> class-1 row; (1,k,2) similar
            if sv.ndim == 3:
                sv = sv[0, :, 1] if sv.shape[-1] == 2 else sv[0, :, -1]
            elif sv.ndim == 2 and sv.shape[0] == 2:
                sv = sv[0]
            sv = np.asarray(sv).reshape(-1)

        cnty = 0
        for s in TransList[FInds]:
            cind = np.squeeze(np.argwhere(ALLFeats == s))
            if np.size(cind) > 0:
                cind = int(cind)
                val = sv[cnty] if cnty < len(sv) else 0.0
                FinalFeatsMat[nn, cind] = val
                FinalFeats[0, cind] += val
                cnty += 1
    return FinalFeats, FinalFeatsMat


# ------------------------------------------------------------------
# Load the golden feature vectors -> 228-dim patient vectors
# ------------------------------------------------------------------
fv = pd.read_csv(
    os.path.join(GOLD, "SynthPat_Multi_Outputs_FeatureVectors.csv"),
    sep=";", index_col=0,
)
FEAT_NAMES = fv["Feature"].values
patients = [c for c in fv.columns
            if c not in ("Feature", "Feature_Category",
                         "Variable_Category", "Variable")]
PATIENT_X = {}
for p in patients:
    vals = pd.to_numeric(fv[p], errors="coerce").values.astype(float)
    PATIENT_X[p] = vals

print("Loaded %d patients x %d features" % (len(patients), N_FEATS))

# ------------------------------------------------------------------
# 1) Ensemble anatomy: composition + feature-subset diversity
# ------------------------------------------------------------------
from collections import Counter
counts = Counter(ALGO_NAMES)
short = {"XGB": "XGBoost", "RF": "RandomForest",
         "ExtraTreesClassifier": "ExtraTrees",
         "LogisticRegression": "LogisticReg.",
         "SGD": "SGD", "Elastic": "ElasticNet"}
label_map = {"XGB": "XGBoost", "RF": "RandomForest",
             "ExtraTreesClassifier": "ExtraTrees",
             "LogisticRegression": "Logistic\nRegression",
             "SGD": "SGD", "Elastic": "ElasticNet"}
palette = {"XGB": "#d62728", "RF": "#1f77b4",
           "ExtraTreesClassifier": "#2ca02c",
           "LogisticRegression": "#9467bd", "SGD": "#ff7f0e",
           "Elastic": "#8c564b"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("Anatomy of the CLL-TIM ensemble: 28 diverse base learners",
             fontsize=13, fontweight="bold")

# (a) algorithm-type composition
ax = axes[0]
order = ["XGB", "RF", "ExtraTreesClassifier", "SGD", "LogisticRegression"]
order = [o for o in order if o in counts] + \
        [k for k in counts if k not in order]
bars = ax.bar([label_map.get(o, o) for o in order],
              [counts[o] for o in order],
              color=[palette.get(o, "#777777") for o in order],
              edgecolor="black")
ax.tick_params(axis="x", labelsize=8.5)
for lbl in ax.get_xticklabels():
    lbl.set_rotation(18)
    lbl.set_ha("right")
for b, o in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
            str(counts[o]), ha="center", fontweight="bold")
ax.set_ylabel("Number of base learners")
ax.set_title("(a) Ensemble composition", fontsize=10)
ax.set_ylim(0, max(counts.values()) + 1.5)

# (b) per-learner feature-subset sizes
ax = axes[1]
sizes = [len(FEATS_OF_BL[nn][0]) for nn in range(N_Chosen)]
idx = np.argsort(sizes)
cols = [palette.get(ALGO_NAMES[i], "#777777") for i in idx]
ax.barh(range(N_Chosen), [sizes[i] for i in idx], color=cols,
        edgecolor="black", linewidth=0.4)
ax.set_xlabel("Features seen by base learner")
ax.set_ylabel("Base learner (sorted)")
ax.set_title("(b) Each learner sees a different\nfeature subset", fontsize=10)
ax.set_yticks([])

# (c) feature coverage: how many learners use each feature
ax = axes[2]
usage = np.zeros(N_FEATS)
for nn in range(N_Chosen):
    for s in TransList[FEATS_OF_BL[nn][0]]:
        cind = np.squeeze(np.argwhere(ALLFeats == s))
        if np.size(cind) > 0:
            usage[int(cind)] += 1
ax.hist(usage, bins=np.arange(0.5, usage.max() + 1.5, 1),
        color="#1f77b4", edgecolor="black")
ax.set_xlabel("Number of base learners using a feature")
ax.set_ylabel("Number of features")
ax.set_title("(c) Feature overlap across learners", fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIGDIR, "fig_ensemble_anatomy.png"), dpi=170)
plt.close(fig)
print("wrote fig_ensemble_anatomy.png")

# ------------------------------------------------------------------
# 2) Score every patient -> votes + risk + confidence
# ------------------------------------------------------------------
ALL_PROBS = {}
for p in patients:
    ALL_PROBS[p] = predict_all(PATIENT_X[p]).ravel()
    print("  %-45s mean=%.3f  std=%.3f" %
          (p, ALL_PROBS[p].mean(), ALL_PROBS[p].std()))

means = {p: ALL_PROBS[p].mean() for p in patients}
order_p = sorted(patients, key=lambda p: means[p], reverse=True)

fig, ax = plt.subplots(figsize=(11.5, 6.2))
ax.set_title("CLL-TIM ensemble scoring: all 28 base-learner votes per patient",
             fontsize=13, fontweight="bold")

# confidence zones (same thresholds the report uses: <0.28 / >=0.58)
ax.axvspan(0, 0.28, color="#b7d7f0", alpha=0.35)
ax.axvspan(0.28, 0.58, color="#f5c9c9", alpha=0.30)
ax.axvspan(0.58, 1.0, color="#f29d9d", alpha=0.35)
ax.text(0.14, len(order_p) - 0.05, "High-Conf.\nLow-Risk", ha="center",
        va="top", fontsize=8, color="#0d3d63")
ax.text(0.43, len(order_p) - 0.05, "Low-Confidence\nzone", ha="center",
        va="top", fontsize=8, color="#7a2020")
ax.text(0.79, len(order_p) - 0.05, "High-Conf.\nHigh-Risk", ha="center",
        va="top", fontsize=8, color="#7a1010")

for i, p in enumerate(order_p):
    y = len(order_p) - 1 - i
    probs = ALL_PROBS[p]
    jit = rng.uniform(-0.13, 0.13, size=len(probs))
    cols = [palette.get(a, "#555555") for a in ALGO_NAMES]
    ax.scatter(probs, y + jit, s=26, c=cols, alpha=0.85,
               edgecolor="black", linewidth=0.4, zorder=3)
    m = probs.mean()
    ax.plot([m, m], [y - 0.28, y + 0.28], color="black", lw=2.4, zorder=4)
    ax.plot([probs.mean() - probs.std(), probs.mean() + probs.std()],
            [y, y], color="black", lw=1.1, zorder=4)
    lab = p.replace("TestPat01_", "TestPat01\n")
    ax.text(-0.015, y, lab, ha="right", va="center", fontsize=8.5)
    conf = "High-Conf" if (m >= 0.58 or m < 0.28) else "Low-Conf"
    ax.text(1.015, y, "%.2f  (%s)" % (m, conf), ha="left",
            va="center", fontsize=8.5,
            color="#7a1010" if m >= 0.58 else
            ("#0d3d63" if m < 0.28 else "#7a5a10"))

ax.set_xlim(0, 1)
ax.set_ylim(-0.7, len(order_p) + 0.1)
ax.set_yticks([])
ax.set_xlabel("Predicted probability of infection or CLL treatment "
              "within 2 years")
handles = [plt.Line2D([0], [0], marker="o", color="w",
           markerfacecolor=palette[o], markeredgecolor="black",
           markersize=8, label=label_map.get(o, o).replace("\n", " "))
           for o in dict.fromkeys(ALGO_NAMES)]
ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
          bbox_to_anchor=(0.5, -0.18))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_ensemble_votes.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("wrote fig_ensemble_votes.png")

# ------------------------------------------------------------------
if want("shap"):
  # 3) SHAP aggregation -> summary bar + patient x feature heatmap
  # ------------------------------------------------------------------
  SHAP_SUM = {}
  SHAP_MAT = {}
  for p in patients:
      ff, fm = shap_all(PATIENT_X[p])
      SHAP_SUM[p] = ff[0]
      SHAP_MAT[p] = fm
      print("  shap done: %s (top=%.3f)" % (p, np.nanmax(ff)))

  mean_abs = np.mean(np.abs(np.vstack([SHAP_SUM[p] for p in patients])), 0)
  top15 = np.argsort(-mean_abs)[:15]

  fig, axes = plt.subplots(1, 2, figsize=(15, 6.4),
                           gridspec_kw={"width_ratios": [1, 1.15]})
  fig.suptitle("Patient-specific explainability: SHAP attributions "
               "aggregated over all 28 base learners", fontsize=13,
               fontweight="bold")

  ax = axes[0]
  names15 = [str(FEAT_NAMES[i])[:52] for i in top15][::-1]
  vals15 = mean_abs[top15][::-1]
  ax.barh(range(15), vals15, color="#395f8f", edgecolor="black")
  ax.set_yticks(range(15))
  ax.set_yticklabels(names15, fontsize=7.5)
  ax.set_xlabel("Mean |summed SHAP value| across 9 synthetic patients")
  ax.set_title("(a) Global drivers of the CLL-TIM score", fontsize=10)

  ax = axes[1]
  H = np.vstack([SHAP_SUM[p][top15] for p in order_p]).T
  vmax = np.nanmax(np.abs(H))
  im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
  ax.set_xticks(range(len(order_p)))
  ax.set_xticklabels([p.replace("TestPat01_", "T01_") for p in order_p],
                     rotation=35, ha="right", fontsize=7.5)
  ax.set_yticks(range(15))
  ax.set_yticklabels([str(FEAT_NAMES[i])[:48] for i in top15], fontsize=7.5)
  ax.set_title("(b) Per-patient summed SHAP contributions\n"
               "(red = pushes risk up, blue = pushes down)", fontsize=10)
  fig.colorbar(im, ax=ax, fraction=0.04, label="Summed SHAP")

  fig.tight_layout(rect=[0, 0, 1, 0.93])
  fig.savefig(os.path.join(FIGDIR, "fig_shap_summary.png"), dpi=170)
  plt.close(fig)
  print("wrote fig_shap_summary.png")
# ------------------------------------------------------------------
if want("missingness"):
  # 4) Missingness robustness: degrade features -> track risk
  # ------------------------------------------------------------------
  fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  REPS = 12
  miss_curves = {p: np.zeros((REPS, len(fracs))) for p in patients}
  for p in patients:
      base = PATIENT_X[p].copy()
      observed = np.where(~np.isnan(base))[0]
      for fi, f in enumerate(fracs):
          k = int(round(f * len(observed)))
          for r in range(REPS):
              x = base.copy()
              if k > 0:
                  mask = rng.choice(observed, size=k, replace=False)
                  x[mask] = np.nan
              miss_curves[p][r, fi] = predict_all(x).mean()
      print("  missingness done: %s" % p)

  fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
  fig.suptitle("Robustness to missing data: CLL-TIM keeps scoring as "
               "features are removed", fontsize=13, fontweight="bold")

  ax = axes[0]
  cmap = plt.cm.viridis
  for i, p in enumerate(order_p):
      mu = miss_curves[p].mean(0)
      sd = miss_curves[p].std(0)
      ax.plot(fracs * 100, mu, marker="o", ms=4,
              color=cmap(i / max(1, len(order_p) - 1)),
              lw=1.6, label=p.replace("TestPat01_", "T01_"))
      ax.fill_between(fracs * 100, mu - sd, mu + sd,
                      color=cmap(i / max(1, len(order_p) - 1)), alpha=0.10)
  ax.axhline(0.5, color="grey", ls="--", lw=1)
  ax.set_xlabel("Features randomly set to missing (%)")
  ax.set_ylabel("Ensemble mean risk")
  ax.set_title("(a) Risk estimate vs. missingness\n"
               "(lines = 9 synthetic patients, bands = +/-1 SD over "
               "12 random masks)", fontsize=9.5)
  ax.legend(fontsize=6.6, ncol=2, loc="best")
  ax.set_ylim(0, 1)

  ax = axes[1]
  drift = np.array([[miss_curves[p][:, -1].mean() -
                     miss_curves[p][:, 0].mean() for p in order_p],
                    [miss_curves[p][:, 0].mean() for p in order_p]]).T
  ax.barh(range(len(order_p)), drift[:, 0],
          color=["#c0392b" if d > 0 else "#2c6fbb" for d in drift[:, 0]],
          edgecolor="black")
  ax.set_yticks(range(len(order_p)))
  ax.set_yticklabels([p.replace("TestPat01_", "T01_") for p in order_p],
                     fontsize=7.5)
  ax.axvline(0, color="black", lw=0.8)
  ax.set_xlabel("Change in mean risk at 60% features missing")
  ax.set_title("(b) Risk drift under heavy missingness\n"
               "(imputation to column means keeps scores stable)", fontsize=9.5)

  fig.tight_layout(rect=[0, 0, 1, 0.93])
  fig.savefig(os.path.join(FIGDIR, "fig_missingness.png"), dpi=170)
  plt.close(fig)
  print("wrote fig_missingness.png")
# ------------------------------------------------------------------
# 5) Pipeline architecture schematic
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")
ax.set_title("CLL-TIM Mobile: from a patient EHR extract to a personalised "
             "risk report", fontsize=14, fontweight="bold", pad=14)


def box(x, y, w, h, text, fc, fs=8.5, bold=False, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, color=(0.2, 0.2, 0.2), lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=13, color=color,
                 lw=lw, linestyle=ls, zorder=5))


# column 1: inputs
box(2, 40, 18, 15, "Patient file\n(1 CSV per patient)\n\nbaseline vars\n"
    "lab history (7y)\ninfection history\nmedication",
    "#eaf2fb", bold=True)
box(2, 20, 18, 14, "CLL-TIM model pack\n\n28 fitted learners (.sav)\n"
    "per-learner feature sets\nper-learner scalers + means\n"
    "k-means train summaries\n(SHAP background)", "#efe9f7", bold=True)

# column 2: feature construction
box(27, 40, 20, 15, "Feature construction\n\n228-feature vector\n"
    "rolling stats over labs\n(min/max/mean/slope,\nquad-fit coefs,\ncounts)\n"
    "baseline + infection flags", "#fff4e0", bold=True)

# column 3: ensemble
box(53, 30, 22, 25, "ENSEMBLE OF 28\nBASE LEARNERS\n\n"
    "13 x XGBoost\n7 x RandomForest\n4 x ExtraTrees\n2 x SGD\n"
    "2 x LogisticRegression\n\neach sees a different\n"
    "feature subset + its own\nimputation/scaling", "#fdeaea", bold=True)

# column 4: aggregation
box(80, 40, 18, 15, "Aggregation\n\nrisk = mean of 28\npredict_proba\n\n"
    "confidence bands:\n<0.28 or >=0.58\n-> High-Confidence",
    "#e9f7ef", bold=True)
box(80, 18, 18, 16, "SHAP per learner\n(Tree / Kernel)\n\nsummed back to\n"
    "the 228 features\n-> top-5 patient-\nspecific risk factors",
    "#e9f7ef", bold=True)

arrow(20, 47.5, 27, 47.5)
arrow(11, 34, 11, 40, color=(0.33, 0.33, 0.33))
arrow(20, 27, 47, 30)
arrow(47, 47.5, 53, 44)
arrow(64, 55, 64, 30, color=(0.6, 0.6, 0.6), ls=":")
arrow(75, 44, 80, 46)
arrow(75, 36, 80, 28)
arrow(89, 40, 89, 34.5, color=(0.53, 0.53, 0.53))

# outputs
box(80, 6, 18, 7, "Patient report\nrisk + confidence +\ntop-5 risk factors",
    "#d5e8d4", bold=True)
arrow(89, 18, 89, 13.2, color="#2c7a3f")

# annotation footer
ax.text(50, 1.2, "Ensemble = 28 independently trained models on "
        "different CV partitions, each with its own feature subset "
        "-> variance reduction + graceful behaviour under missing data",
        ha="center", fontsize=8, style="italic", color="#444444")

fig.savefig(os.path.join(FIGDIR, "fig_pipeline.png"), dpi=170,
            bbox_inches="tight")
plt.close(fig)
print("wrote fig_pipeline.png")

# ------------------------------------------------------------------
# save per-patient vote table for README/reference
# ------------------------------------------------------------------
tbl = pd.DataFrame(
    {p: ALL_PROBS[p] for p in order_p},
    index=["BL%02d_%s" % (n, ALGO_NAMES[n]) for n in range(N_Chosen)]
)
tbl.loc["ENSEMBLE_MEAN"] = tbl.mean(0)
tbl.loc["ENSEMBLE_STD"] = tbl.loc[[i for i in tbl.index
                                   if i.startswith("BL")]].std(0)
tbl.to_csv(os.path.join(FIGDIR, "ensemble_votes.csv"))
print("wrote ensemble_votes.csv")
print("ALL DONE")
