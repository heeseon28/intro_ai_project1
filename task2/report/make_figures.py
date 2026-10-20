"""Figures and numbers for the Activity 2 report, read only from the committed Task 2 results.

Usage (from anywhere): python3 task2/report/make_figures.py [VMMRdb root]
The image root is only needed for the misclassified-example strip (errors.pdf).
"""
import csv, json, sys, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "result_task2"
RUN_A = RES / "baseline_seed0_DonghyunShin"   # GPU run, numbers quoted in the report
RUN_B = RES / "baseline_seed0_Hyeonil_Cho"     # independent CPU re-run, used for reproducibility
VMMR = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/dataset/VMMRdb").expanduser()
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE, ORANGE, GRAY = "#2F6DB5", "#D9822B", "#8C8C8C"
MAKE = {"chevrolet": "Chevy", "volkswagen": "VW", "bmw": "BMW", "gmc": "GMC"}
NICE = {"silverado": "Silverado", "sierra_1500": "Sierra 1500", "ram_1500": "Ram 1500", "ram_2500": "Ram 2500",
        "civic": "Civic", "accord": "Accord", "gti": "GTI", "golf": "Golf", "corvette": "Corvette",
        "mustang": "Mustang", "impala": "Impala"}
nice = lambda n: NICE.get(n.split("/")[1], n.split("/")[1])

def read(run, name):
    return list(csv.DictReader(open(run / name)))

# ---------- learning curves (run A, with run B's validation curve for reproducibility)
ha, hb = read(RUN_A, "history.csv"), read(RUN_B, "history.csv")
ep = [int(r["epoch"]) for r in ha]
g = lambda h, k: [float(r[k]) for r in h]
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.6))
ax[0].plot(ep, g(ha, "train_loss"), color=BLUE, label="train (run A)")
ax[0].plot(ep, g(ha, "val_loss"), color=ORANGE, label="validation (run A)")
ax[0].plot(ep, g(hb, "val_loss"), color=GRAY, lw=1, ls="-.", label="validation (run B)")
ax[0].axvline(5, color="gray", lw=0.6, ls=":")
ax[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
ax[1].plot(ep, g(ha, "train_accuracy"), color=BLUE)
ax[1].plot(ep, g(ha, "val_accuracy"), color=ORANGE)
ax[1].plot(ep, g(hb, "val_accuracy"), color=GRAY, lw=1, ls="-.")
ax[1].plot(ep, g(ha, "val_macro_f1"), color=ORANGE, ls="--", label="val macro-F1 (run A)")
ax[1].set(xlabel="epoch", title="Accuracy / macro-F1", ylim=(0.0, 1.0))
for a in ax:
    a.set_xticks([1, 4, 8, 12]); a.grid(alpha=0.3, lw=0.4)
handles = ax[0].get_legend_handles_labels()[0] + ax[1].get_legend_handles_labels()[0]
fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.13), fontsize=6.3, handlelength=1.8, columnspacing=1.0)
fig.tight_layout(w_pad=0.8, rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "curves.pdf"); plt.close(fig)

# ---------- confusion matrix: full (make blocks) + zoom on the look-alike classes
rows = list(csv.reader(open(RUN_A / "confusion_matrix_counts.csv")))
names = [r[0] for r in rows[1:]]
M = np.array([[int(x) for x in r[1:]] for r in rows[1:]])
assert M.shape == (50, 50) and names == rows[0][1:]
R = M / M.sum(1, keepdims=True)
makes = [n.split("/")[0] for n in names]
bounds = [i for i in range(1, 50) if makes[i] != makes[i - 1]]
starts = [0] + bounds

fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.05), gridspec_kw={"width_ratios": [1, 1.02]})
im = ax[0].imshow(R, cmap="Blues", vmin=0, vmax=1)
for b in bounds:
    ax[0].axhline(b - 0.5, color="k", lw=0.35); ax[0].axvline(b - 0.5, color="k", lw=0.35)
mid = [(s + e - 1) / 2 for s, e in zip(starts, starts[1:] + [50])]
lab = [MAKE.get(makes[s], makes[s].title()) for s in starts]
ax[0].set_xticks(mid, lab, rotation=90); ax[0].set_yticks(mid, lab)
ax[0].tick_params(length=0)
ax[0].set(xlabel="predicted class (grouped by make)", ylabel="true class", title="(a) All 50 classes, row-normalized")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)

hard = ["chevrolet/silverado", "gmc/sierra_1500", "dodge/ram_1500", "dodge/ram_2500", "honda/civic",
        "honda/accord", "bmw/325i", "bmw/328i", "bmw/528i", "volkswagen/gti", "volkswagen/golf"]
idx = [names.index(n) for n in hard]
Z = R[np.ix_(idx, idx)]
ax[1].imshow(Z, cmap="Blues", vmin=0, vmax=1)
for i in range(len(idx)):
    for j in range(len(idx)):
        if Z[i, j] >= 0.02:
            ax[1].text(j, i, f"{Z[i, j] * 100:.0f}", ha="center", va="center", fontsize=6,
                       color="white" if Z[i, j] > 0.5 else "black")
short = [nice(n) for n in hard]
ax[1].set_xticks(range(len(idx)), short, rotation=90); ax[1].set_yticks(range(len(idx)), short)
ax[1].set(title="(b) Look-alike classes: % of true class predicted as column")
fig.tight_layout(w_pad=1.2)
fig.savefig(FIG / "confusion.pdf"); plt.close(fig)

# ---------- numbers quoted in the report (printed, then copied into main.tex)
for run in (RUN_A, RUN_B):
    m = json.load(open(run / "metrics.json")); t, v = m["train_at_best_checkpoint"], m["validation_at_best_checkpoint"]
    print(f"{run.name}: best epoch {m['best_epoch']}  train acc {t['accuracy']:.4f} loss {t['loss']:.3f}  "
          f"val acc {v['accuracy']:.4f} mF1 {v['macro_f1']:.4f} bal {v['balanced_accuracy']:.4f} loss {v['loss']:.3f}  "
          f"gap {(t['accuracy'] - v['accuracy']) * 100:.2f}pp  acc-mF1 {(v['accuracy'] - v['macro_f1']) * 100:.2f}pp")
tot = M.sum(); err = tot - np.trace(M)
cross = sum(M[i, j] for i in range(50) for j in range(50) if i != j and makes[i] != makes[j])
pc = read(RUN_A, "per_class_metrics.csv"); info = {r["make_model"]: r for r in read(RUN_A, "class_info.csv")}
f1 = np.array([float(r["f1"]) for r in pc]); ntr = np.array([int(info[r["make_model"]]["n_train"]) for r in pc])
f1b = {r["make_model"]: float(r["f1"]) for r in read(RUN_B, "per_class_metrics.csv")}
pairs = sorted(((M[i, j], names[i], names[j]) for i in range(50) for j in range(50) if i != j), reverse=True)[:8]
pred = read(RUN_A, "validation_predictions.csv")
dec = collections.defaultdict(lambda: [0, 0])
for r in pred:
    y = int(r["rel_path"].split("/")[0].rsplit("_", 1)[1])
    d = "<1990" if y < 1990 else "1990s" if y < 2000 else "2000s" if y < 2010 else ">=2010"
    dec[d][0] += r["correct"] == "True"; dec[d][1] += 1
print(f"val error {err / tot:.4f}  cross-make {cross / tot:.4f}  within-make {(err - cross) / tot:.4f}  cross share {cross / err:.3f}")
print(f"F1 median {np.median(f1):.3f} min {f1.min():.3f} max {f1.max():.3f}  corr(log n_train, F1) {np.corrcoef(np.log(ntr), f1)[0, 1]:.2f}")
print(f"per-class F1 corr run A vs run B {np.corrcoef(f1, [f1b[r['make_model']] for r in pc])[0, 1]:.3f}")
print("top pairs", pairs)
print("by decade", {k: (v[1], round(v[0] / v[1], 3)) for k, v in dec.items()})

# ---------- misclassified validation images for the main confusion pairs (first in file order, not cherry-picked)
want = [("gmc/sierra_1500", "chevrolet/silverado"), ("dodge/ram_1500", "dodge/ram_2500"), ("honda/civic", "honda/accord"),
        ("bmw/328i", "bmw/325i"), ("chevrolet/corvette", "ford/mustang"), ("ford/mustang", "chevrolet/impala")]
picks = [next(r for r in pred if r["make_model"] == t and r["prediction_label"] == p) for t, p in want]
if VMMR.is_dir():
    fig, ax = plt.subplots(1, len(picks), figsize=(7.1, 1.3))
    for a, r in zip(ax, picks):
        with Image.open(VMMR / r["rel_path"]) as im:
            im = im.convert("RGB")
            w, h = im.size; cw, ch = min(w, h * 4 // 3), min(h, w * 3 // 4)  # same 4:3 crop for every tile
            im = im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)).resize((320, 240))
            a.imshow(im)
        yr = r["rel_path"].split("/")[0].rsplit("_", 1)[1]
        a.set_title(f"true: {nice(r['make_model'])} ({yr})\npred: {nice(r['prediction_label'])}", fontsize=6.5)
        a.axis("off")
    fig.tight_layout(w_pad=0.3)
    fig.savefig(FIG / "errors.pdf", dpi=200); plt.close(fig)
else:
    print("VMMRdb not found; kept the existing errors.pdf")
