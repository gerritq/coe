import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

DATASET_GROUPS = {
    "drlDomain": [
        "drlDomain_arxiv",
        "drlDomain_writing_prompt",
        "drlDomain_yelp_review",
        "drlDomain_xsum",
    ],
    "drlAttack": [
        "drlAttack_multi_llm_mixing",
        "drlAttack_paraphrase_attacks_llm",
        "drlAttack_perturbation_attacks_llm",
        "drlAttack_prompt_attacks_llm",
    ],
    "multisocial": [
        "multisocial_en",
        "multisocial_de",
        "multisocial_ru",
        "multisocial_zh",
    ],
    "tsm": [
        "tsm_first",
        "tsm_extend",
        "tsm_sums",
        "tsm_tst",
    ],
    "m4": [
        "m4_bloomz",
        "m4_cohere",
        "m4_dolly",
        "m4_gpt4",
    ],
}

FAMILY_ORDER = ["drlDomain", "drlAttack", "multisocial", "tsm", "m4"]

METHOD_SPECS = {
    "probe_default": {"kind": "probe", "mode": "default", "label": "Probe (default)"},
    "encoder": {"kind": "baseline", "model": "encoder", "label": "Encoder"},
    "text_fluoroscopy": {"kind": "baseline", "model": "text_fluoroscopy", "label": "Fluoroscopy"},
    "biscope": {"kind": "baseline", "model": "biscope", "label": "BiScope"},
    "repreguard": {"kind": "baseline", "model": "repreguard", "label": "RepreGuard"},
}


def _dataset_to_family() -> dict[str, str]:
    out = {}
    for fam, ds_list in DATASET_GROUPS.items():
        for ds in ds_list:
            out[ds] = fam
    return out


def _short_label(dataset_name: str) -> str:
    if dataset_name.startswith("drlDomain_"):
        return dataset_name.replace("drlDomain_", "")
    if dataset_name.startswith("drlAttack_"):
        return dataset_name.replace("drlAttack_", "").replace("_attacks_llm", "").replace("_", "-")
    if dataset_name.startswith("multisocial_"):
        return dataset_name.replace("multisocial_", "")
    if dataset_name.startswith("tsm_"):
        return dataset_name.replace("tsm_", "")
    if dataset_name.startswith("m4_"):
        return dataset_name.replace("m4_", "")
    return dataset_name


def _probe_auroc(test_metrics: dict) -> float | None:
    wm = test_metrics.get("weighted_projection_metrics", {})
    if "auroc" in wm:
        return wm.get("auroc")
    mm = test_metrics.get("mean_projection_metrics", {})
    if "auroc" in mm:
        return mm.get("auroc")
    return None


def _collect_method_entries(method_key: str) -> dict[str, dict[str, float]]:
    spec = METHOD_SPECS[method_key]
    ds_to_family = _dataset_to_family()

    # family -> train_ds -> test_ds -> auroc
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    if spec["kind"] == "probe":
        for filename in sorted(os.listdir(PROBE_DIR)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(PROBE_DIR, filename)
            with open(path, "r") as f:
                obj = json.load(f)

            args = obj.get("args", {})
            if args.get("mode") != spec["mode"]:
                continue

            train_ds = args.get("dataset")
            test_ds = args.get("target_dataset")
            if train_ds is None or test_ds is None:
                continue
            if train_ds == test_ds:
                continue  # OOD only

            fam_train = ds_to_family.get(train_ds)
            fam_test = ds_to_family.get(test_ds)
            if fam_train is None or fam_train != fam_test:
                continue

            auroc = _probe_auroc(obj.get("test_metrics", {}))
            if auroc is None:
                continue
            out[fam_train][train_ds][test_ds] = float(auroc)

    else:
        for filename in sorted(os.listdir(BASELINE_DIR)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(BASELINE_DIR, filename)
            with open(path, "r") as f:
                obj = json.load(f)

            args = obj.get("args", {})
            if args.get("model") != spec["model"]:
                continue

            train_ds = args.get("dataset")
            test_ds = args.get("target_dataset")
            if train_ds is None or test_ds is None:
                continue
            if train_ds == test_ds:
                continue  # OOD only

            fam_train = ds_to_family.get(train_ds)
            fam_test = ds_to_family.get(test_ds)
            if fam_train is None or fam_train != fam_test:
                continue

            auroc = obj.get("metrics", {}).get("auroc")
            if auroc is None:
                continue
            # RepreGuard reports AUROC on a 0-100 scale in this codebase.
            if spec["model"] == "repreguard" and auroc > 1.0:
                auroc = auroc / 100.0
            out[fam_train][train_ds][test_ds] = float(auroc)

    return out


def _build_family_matrix(family: str, entries: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    labels = DATASET_GROUPS[family]
    idx = {d: i for i, d in enumerate(labels)}
    mat = np.full((len(labels), len(labels)), np.nan, dtype=float)

    for src, tgt_map in entries.items():
        if src not in idx:
            continue
        for tgt, val in tgt_map.items():
            if tgt not in idx:
                continue
            mat[idx[src], idx[tgt]] = val

    return labels, mat


def _plot_family_matrix(ax, family: str, entries: dict[str, dict[str, float]], cmap, vmin=0.0, vmax=1.0):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    labels, mat = _build_family_matrix(family, entries)
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    short_labels = [_short_label(x) for x in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Test", fontsize=8)
    ax.set_ylabel("Train", fontsize=8)
    ax.set_title(family, fontsize=9)
    for rr in range(mat.shape[0]):
        for cc in range(mat.shape[1]):
            v = mat[rr, cc]
            if np.isnan(v):
                continue
            txt_color = "white" if v < 0.55 else "black"
            ax.text(cc, rr, f"{v:.2f}", ha="center", va="center", fontsize=6, color=txt_color)
    return im


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ood_figure.pdf")

    entries_fluo = _collect_method_entries("text_fluoroscopy")
    entries_biscope = _collect_method_entries("biscope")
    entries_repre = _collect_method_entries("repreguard")
    entries_default = _collect_method_entries("probe_default")
    entries_encoder = _collect_method_entries("encoder")

    panel_methods = [
        ("Fluoroscopy", entries_fluo),
        ("BiScope", entries_biscope),
        ("RepreGuard", entries_repre),
        ("Probe (default)", entries_default),
    ]

    n_families = len(FAMILY_ORDER)
    n_cols = 2 * n_families
    fig, axes = plt.subplots(3, n_cols, figsize=(34, 12), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")

    im = None
    # Row 1: fluo then biscope
    for block_idx, (title, fam_entries) in enumerate(panel_methods[:2]):
        col_offset = block_idx * n_families
        for i, family in enumerate(FAMILY_ORDER):
            ax = axes[0, col_offset + i]
            im = _plot_family_matrix(ax, family, fam_entries.get(family, {}), cmap)
            if i == 0:
                ax.text(
                    -0.45,
                    1.18,
                    title,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="bottom",
                )

    # Row 2: repreguard then default probe
    for block_idx, (title, fam_entries) in enumerate(panel_methods[2:]):
        col_offset = block_idx * n_families
        for i, family in enumerate(FAMILY_ORDER):
            ax = axes[1, col_offset + i]
            im = _plot_family_matrix(ax, family, fam_entries.get(family, {}), cmap)
            if i == 0:
                ax.text(
                    -0.45,
                    1.18,
                    title,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="bottom",
                )

    # Row 3: encoder in first block, second block left empty.
    for i, family in enumerate(FAMILY_ORDER):
        ax = axes[2, i]
        im = _plot_family_matrix(ax, family, entries_encoder.get(family, {}), cmap)
        if i == 0:
            ax.text(
                -0.45,
                1.18,
                "Encoder",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="bottom",
            )
    for j in range(n_families, n_cols):
        axes[2, j].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.01)
        cbar.set_label("AUROC")

    fig.suptitle("OOD Confusion Matrices (4x4 per family)", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
