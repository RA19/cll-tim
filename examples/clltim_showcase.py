"""Illustrate the 28 learner outputs on bundled synthetic feature vectors.

Run with the pinned environment: python examples/clltim_showcase.py
This replays inference, not feature extraction. It produces a votes figure and
CSV, with no clinical performance or SHAP correctness claims.
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "CLLTIMData"


def load_pickle(name):
    with (DATA / name).open("rb") as handle:
        return pickle.load(handle)


def learner_outputs():
    chosen = np.load(str(DATA / "_28_ToKeep.npy"))
    cv = np.load(str(DATA / "_Collect_CVData.npy"))[chosen, :]
    model_names = np.load(str(DATA / "_Collect_ModelNames.npy"))[chosen, :]
    indices = np.load(str(DATA / "CLL_TIM_FeatIndexes.npy"), allow_pickle=True).item()
    union = np.unique(np.concatenate([indices[row[0]] for row in cv]))
    means = load_pickle("vecgetcolmean.pkl")
    scalers = load_pickle("vec_scaler.pkl")
    features = pd.read_csv(
        str(ROOT / "docs/synthetic_patient_examples/SynthPat_Multi_Outputs_FeatureVectors.csv"),
        sep=";", index_col=0,
    )
    metadata = {"Feature", "Feature_Category", "Variable_Category", "Variable"}
    patients = [column for column in features.columns if column not in metadata]
    matrix = features[patients].to_numpy(dtype=float).T
    outputs = np.empty((len(chosen), len(patients)))
    families = [str(row[1]) for row in model_names]
    for i in range(len(chosen)):
        model = load_pickle("CLL-TIM_BLN_%d.sav" % i)
        subset = indices[cv[i, 0]]
        positions = [int(np.where(union == feature)[0][0]) for feature in subset]
        x = matrix[:, positions].copy()
        if families[i] != "XGB":
            row, col = np.where(np.isnan(x))
            x[row, col] = np.asarray(means[i])[col]
            x = scalers[i].transform(x)
        outputs[i] = model.predict_proba(x)[:, 1]
    return patients, families, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patients, families, outputs = learner_outputs()
    order = np.argsort(-outputs.mean(axis=0))
    patients = [patients[i] for i in order]
    outputs = outputs[:, order]
    palette = {"XGB": "#b76a38", "RF": "#245775",
               "ExtraTreesClassifier": "#168287", "Elastic": "#7d6b9e",
               "LogisticRegression": "#6c7b88"}
    labels = {"XGB": "XGBoost", "RF": "Random Forest",
              "ExtraTreesClassifier": "Extra Trees", "Elastic": "Elastic-net",
              "LogisticRegression": "Logistic Regression"}
    rng = np.random.RandomState(42)
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.set_title("CLL-TIM learner outputs | nine synthetic patients", fontsize=14, pad=32)
    for low, high, color, label in (
        (0, .28, "#dcecf4", "High confidence / low risk"),
        (.28, .58, "#f0f1f3", "Low confidence"),
        (.58, 1, "#fae7da", "High confidence / high risk"),
    ):
        ax.axvspan(low, high, color=color)
        ax.text((low + high) / 2, 1.015, label, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9)
    colors = [palette[family] for family in families]
    for i, patient in enumerate(patients):
        y = len(patients) - 1 - i
        values = outputs[:, i]
        mean, sd = values.mean(), values.std()
        ax.scatter(values, y + rng.uniform(-.12, .12, len(values)), c=colors,
                   s=27, edgecolors="white", linewidths=.35, zorder=3)
        ax.plot([mean, mean], [y - .26, y + .26], color="#142e46", lw=2, zorder=4)
        ax.plot([mean - sd, mean + sd], [y, y], color="#142e46", lw=1, zorder=2)
        score = float(np.round(mean, 2))
        confidence = "HC" if score < .28 or score >= .58 else "LC"
        ax.text(1.015, y, "%.2f / %s" % (score, confidence), va="center", fontsize=9)
    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels([name.replace("TestPat01_", "TestPat01\n")
                        for name in reversed(patients)], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-.6, len(patients) - .4)
    ax.set_xlabel("Component-model output / ensemble score", labelpad=10)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=palette[family],
                          label=labels[family]) for family in dict.fromkeys(families)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.12),
              ncol=5, frameon=False, fontsize=8)
    fig.text(.5, .015, "Synthetic illustration | vertical marks: means | bars: +/-1 SD"
             " | confidence uses the rounded mean, not SD", ha="center", fontsize=9,
             color="#4e6575")
    fig.subplots_adjust(left=.34, right=.87, top=.85, bottom=.19)
    fig.savefig(str(args.output_dir / "fig_ensemble_votes.png"), dpi=170)
    plt.close(fig)
    table = pd.DataFrame(outputs, columns=patients,
                         index=["BL%02d_%s" % (i, family) for i, family in enumerate(families)])
    table.loc["ENSEMBLE_MEAN"] = outputs.mean(axis=0)
    table.loc["ENSEMBLE_STD"] = outputs.std(axis=0)
    table.to_csv(str(args.output_dir / "ensemble_votes.csv"))
    print("Wrote synthetic vote figure and CSV to " + os.fspath(args.output_dir))


if __name__ == "__main__":
    main()
