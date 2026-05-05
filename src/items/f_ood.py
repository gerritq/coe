import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

FAMILIES = ["detectrl", "multisocial", "tsm"]


def _family(dataset_name: str) -> str | None:
    if dataset_name.startswith("detectrl_"):
        return "detectrl"
    if dataset_name.startswith("multisocial_"):
        return "multisocial"
    if dataset_name.startswith("tsm_"):
        return "tsm"
    return None


def _short_label(dataset_name: str) -> str:
    fam = _family(dataset_name)
    if fam == "detectrl":
        return dataset_name.replace("detectrl_", "")
    if fam == "multisocial":
        return dataset_name.replace("multisocial_", "")
    if fam == "tsm":
        # tsm_paras_en -> paras-en, tsm_sums_pt -> sums-pt
        return dataset_name.replace("tsm_", "").replace("_", "-")
    return dataset_name


def _extract_auroc(obj: dict) -> float | None:
    tm = obj.get("test_metrics", {})
    if "meta_metrics" in tm:
        return tm["meta_metrics"].get("auroc")
    if "weighted_projection_metrics" in tm:
        return tm["weighted_projection_metrics"].get("auroc")
    if "mean_projection_metrics" in tm:
        return tm["mean_projection_metrics"].get("auroc")
    return None


def _parse_filename(filename: str) -> tuple[str, str, str] | None:
    # Example: default_last_token_detectrl_arxiv_2_detectrl_writing_prompt.json
    m = re.match(r"^(.*?)_last_token_(.+)_2_(.+)\.json$", filename)
    if not m:
        return None
    mode = m.group(1)
    src = m.group(2)
    tgt = m.group(3)
    return mode, src, tgt


def load_probe_ood_aurocs() -> dict[str, dict[str, dict[str, float]]]:
    # mode -> src -> tgt -> auroc
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue
        parsed = _parse_filename(filename)
        if parsed is None:
            continue
        mode, src, tgt = parsed
        if _family(src) != _family(tgt):
            continue
        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r") as f:
            obj = json.load(f)
        auroc = _extract_auroc(obj)
        if auroc is None:
            continue
        out[mode][src][tgt] = float(auroc)
    return out


def build_matrices(data: dict[str, dict[str, dict[str, float]]]):
    # mode -> family -> (labels, matrix)
    result: dict[str, dict[str, tuple[list[str], np.ndarray]]] = defaultdict(dict)
    for mode, src_map in data.items():
        by_family: dict[str, set[str]] = defaultdict(set)
        for src, tgt_map in src_map.items():
            fam = _family(src)
            if fam is None:
                continue
            by_family[fam].add(src)
            for tgt in tgt_map:
                by_family[fam].add(tgt)

        for fam in FAMILIES:
            labels = sorted(by_family.get(fam, set()))
            if not labels:
                continue
            idx = {d: i for i, d in enumerate(labels)}
            mat = np.full((len(labels), len(labels)), np.nan, dtype=float)
            for src, tgt_map in src_map.items():
                if _family(src) != fam:
                    continue
                for tgt, val in tgt_map.items():
                    if tgt in idx and src in idx:
                        mat[idx[src], idx[tgt]] = val
            result[mode][fam] = (labels, mat)
    return result


def plot_all_confusions(mats: dict[str, dict[str, tuple[list[str], np.ndarray]]]) -> str:
    modes = sorted(mats.keys())
    families_present = [f for f in FAMILIES if any(f in mats[m] for m in modes)]
    if not modes or not families_present:
        raise RuntimeError("No OOD probe matrices found in output/probe/sandbox.")

    n_rows = len(modes)
    n_cols = len(families_present)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")

    for r, mode in enumerate(modes):
        for c, fam in enumerate(families_present):
            ax = axes[r, c]
            if fam not in mats[mode]:
                ax.axis("off")
                ax.set_title(f"{mode} | {fam}\n(no data)")
                continue
            labels, mat = mats[mode][fam]
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=cmap)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            short_labels = [_short_label(x) for x in labels]
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(short_labels, fontsize=8)
            ax.set_xlabel("Test subset")
            ax.set_ylabel("Train subset")
            ax.set_title(f"mode={mode} | {fam}")

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if np.isnan(v):
                        continue
                    txt_color = "white" if v < 0.55 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=txt_color)

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.92, pad=0.02)
    cbar.set_label("AUROC")
    fig.suptitle("Probe AUROC Matrices (including diagonal)", y=0.995, fontsize=13)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ood_probe_confusion_matrices.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def save_auroc_report(mats: dict[str, dict[str, tuple[list[str], np.ndarray]]]) -> str:
    report = {}
    for mode, fam_map in mats.items():
        report[mode] = {}
        for fam, (_, mat) in fam_map.items():
            vals = mat[~np.isnan(mat)]
            report[mode][fam] = {
                "n_pairs": int(vals.size),
                "mean_auroc": float(np.mean(vals)) if vals.size else None,
                "min_auroc": float(np.min(vals)) if vals.size else None,
                "max_auroc": float(np.max(vals)) if vals.size else None,
            }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "ood_probe_auroc_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    return out_json


def main() -> None:
    data = load_probe_ood_aurocs()
    mats = build_matrices(data)
    fig_path = plot_all_confusions(mats)
    rep_path = save_auroc_report(mats)
    print(f"Saved figure: {fig_path}")
    print(f"Saved AUROC report: {rep_path}")


if __name__ == "__main__":
    main()
