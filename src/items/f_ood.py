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
    "raid": [
        "raid_cohere_chat",
        "raid_gpt4",
        "raid_llama_chat",
        "raid_mistral_chat",
    ],
}

FAMILY_ORDER = ["drlDomain", "multisocial", "tsm", "raid"]
FAMILY_LABELS = {
    "drlDomain": "DetectRL",
    "multisocial": "Multisocial",
    "tsm": "TSM",
    "raid": "RAID",
}

METHOD_SPECS = {
    "probe_default": {"kind": "probe", "mode": "default"},
    "probe_meta_no_pca": {"kind": "probe", "mode": "meta_no_pca"},
    "encoder": {"kind": "baseline", "model": "encoder"},
    "text_fluoroscopy": {"kind": "baseline", "model": "text_fluoroscopy"},
    "biscope": {"kind": "baseline", "model": "biscope"},
    "repreguard": {"kind": "baseline", "model": "repreguard"},
}

DATASET_ALIASES = {
    "raidModel_cohere_chat": "raid_cohere_chat",
    "raidModel_gpt4": "raid_gpt4",
    "raidModel_llama_chat": "raid_llama_chat",
    "raidModel_mistral_chat": "raid_mistral_chat",
}


def _dataset_to_family() -> dict[str, str]:
    out = {}
    for fam, ds_list in DATASET_GROUPS.items():
        for ds in ds_list:
            out[ds] = fam
    return out


def _canonical_dataset_name(ds: str | None) -> str | None:
    if ds is None:
        return None
    return DATASET_ALIASES.get(ds, ds)


def _short_label(dataset_name: str) -> str:
    if dataset_name.startswith("drlDomain_"):
        label = dataset_name.replace("drlDomain_", "")
        if label == "writing_prompt":
            return "Reddit"
        if label == "yelp_review":
            return "yelp"
        return label
    if dataset_name.startswith("multisocial_"):
        return dataset_name.replace("multisocial_", "")
    if dataset_name.startswith("tsm_"):
        return dataset_name.replace("tsm_", "")
    if dataset_name.startswith("raid_"):
        if dataset_name == "raid_cohere_chat":
            return "cohere"
        if dataset_name == "raid_gpt4":
            return "gpt4"
        if dataset_name == "raid_llama_chat":
            return "llama"
        if dataset_name == "raid_mistral_chat":
            return "mistral"
        return dataset_name.replace("raid_", "")
    return dataset_name


def _probe_auroc(test_metrics: dict) -> float | None:
    meta = test_metrics.get("meta_metrics", {})
    if "auroc" in meta:
        return meta.get("auroc")
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

            train_ds = _canonical_dataset_name(args.get("dataset"))
            test_ds = _canonical_dataset_name(args.get("target_dataset"))
            if train_ds is None or test_ds is None:
                continue

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

            train_ds = _canonical_dataset_name(args.get("dataset"))
            test_ds = _canonical_dataset_name(args.get("target_dataset"))
            if train_ds is None or test_ds is None:
                continue

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


def _plot_family_matrix(
    ax,
    family: str,
    entries: dict[str, dict[str, float]],
    vmin: float = 0.0,
    vmax: float = 1.0,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xticklabels: bool = True,
):
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="#f0f0f0")
    labels, mat = _build_family_matrix(family, entries)
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    short_labels = [_short_label(x) for x in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    if show_xticklabels:
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=17)
    else:
        ax.set_xticklabels([])
    ax.set_yticklabels(short_labels, fontsize=17)
    ax.set_xlabel("Test" if show_xlabel else "", fontsize=18)
    ax.set_ylabel("Train" if show_ylabel else "", fontsize=18)
    ax.set_title(FAMILY_LABELS.get(family, family), fontsize=20)
    for rr in range(mat.shape[0]):
        for cc in range(mat.shape[1]):
            v = mat[rr, cc]
            if np.isnan(v):
                continue
            denom = max(vmax - vmin, 1e-12)
            norm_v = (v - vmin) / denom
            txt_color = "white" if norm_v >= 0.6 else "black"
            label = f"{v:.2f}"
            if label.startswith("0"):
                label = label[1:]
            elif label.startswith("-0"):
                label = "-" + label[2:]
            ax.text(cc, rr, label, ha="center", va="center", fontsize=20, color=txt_color)
    return im


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ood_figure.pdf")

    panel_methods = [
        ("TextFluoroscopy", _collect_method_entries("text_fluoroscopy")),
        ("BiScope", _collect_method_entries("biscope")),
        ("RepreGuard", _collect_method_entries("repreguard")),
        ("RoBERTa", _collect_method_entries("encoder")),
        ("LLP", _collect_method_entries("probe_default")),
        ("CLP", _collect_method_entries("probe_meta_no_pca")),
    ]

    # Relative color scale from observed AUROC values only.
    observed_vals = []
    for _, fam_entries in panel_methods:
        for fam in FAMILY_ORDER:
            for _, tgt_map in fam_entries.get(fam, {}).items():
                for val in tgt_map.values():
                    observed_vals.append(float(val))
    if observed_vals:
        global_vmin = min(observed_vals)
        global_vmax = max(observed_vals)
        if global_vmax <= global_vmin:
            global_vmax = global_vmin + 1e-6
    else:
        global_vmin, global_vmax = 0.0, 1.0

    n_families = len(FAMILY_ORDER)
    n_cols = 2 * n_families
    n_rows = (len(panel_methods) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(44, 6 * n_rows), squeeze=False, constrained_layout=True)
    im = None
    block_specs: list[tuple[int, int, str]] = []
    for panel_idx, (title, fam_entries) in enumerate(panel_methods):
        row_idx = panel_idx // 2
        block_idx = panel_idx % 2
        col_offset = block_idx * n_families
        block_specs.append((row_idx, col_offset, title))
        for i, family in enumerate(FAMILY_ORDER):
            ax = axes[row_idx, col_offset + i]
            show_xlabel = row_idx == (n_rows - 1)
            show_ylabel = row_idx == (n_rows - 1) and (col_offset + i) == 0
            show_xticklabels = row_idx == (n_rows - 1)
            im = _plot_family_matrix(
                ax,
                family,
                fam_entries.get(family, {}),
                vmin=global_vmin,
                vmax=global_vmax,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                show_xticklabels=show_xticklabels,
            )

    # Hide any unused blocks on the last row.
    total_blocks = n_rows * 2
    unused_blocks = total_blocks - len(panel_methods)
    if unused_blocks > 0:
        for b in range(total_blocks - unused_blocks, total_blocks):
            row_idx = b // 2
            block_idx = b % 2
            col_offset = block_idx * n_families
            for j in range(col_offset, col_offset + n_families):
                axes[row_idx, j].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location="right", shrink=0.9, pad=0.02)
        cbar.set_label("AUROC", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    for ax in axes.ravel():
        ax.set_aspect("equal")
    fig.canvas.draw()
    for row_idx, col_offset, title in block_specs:
        block_axes = [axes[row_idx, col_offset + i] for i in range(n_families)]
        x0 = min(ax.get_position().x0 for ax in block_axes)
        x1 = max(ax.get_position().x1 for ax in block_axes)
        y1 = max(ax.get_position().y1 for ax in block_axes)
        fig.text(
            (x0 + x1) / 2,
            y1 + 0.02,
            title,
            ha="center",
            va="bottom",
            fontsize=24,
            fontweight="bold",
        )

    # Left-side group labels: Baselines (first two rows) and Linear Probes (last row).
    n_rows_actual = axes.shape[0]
    if n_rows_actual >= 3:
        row0 = axes[0, 0].get_position()
        row1 = axes[1, 0].get_position()
        row2 = axes[2, 0].get_position()

        y_baselines = 0.5 * (row0.y0 + row0.y1 + row1.y0 + row1.y1) / 2
        y_linear = 0.5 * (row2.y0 + row2.y1)
        x_label = max(0.005, min(row0.x0, row1.x0, row2.x0) - 0.055)

        fig.text(
            x_label,
            y_baselines,
            "Baselines",
            rotation=90,
            va="center",
            ha="center",
            fontsize=22,
            fontweight="bold",
        )
        fig.text(
            x_label,
            y_linear,
            "Linear Probes",
            rotation=90,
            va="center",
            ha="center",
            fontsize=22,
            fontweight="bold",
        )

    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
