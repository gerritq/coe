import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

MODE = "default"
FAMILIES = ["drlDomain", "multisocial", "tsm", "raidModel"]
FAMILY_LABELS = {
    "drlDomain": "DetectRL (Domains)",
    "multisocial": "Multisocial (Languages)",
    "tsm": "TSM (Tasks)",
    "raidModel": "RAID (Generators)",
}
COLORS = {
    "drlDomain": "#1f77b4",
    "multisocial": "#d62728",
    "tsm": "#2ca02c",
    "raidModel": "#ff7f0e",
}
MARKERS = {
    "drlDomain": "o",
    "multisocial": "s",
    "tsm": "^",
    "raidModel": "D",
}
FONT_SIZES = {
    "axis": 16,
    "ticks": 14,
    "legend": 14,
}


def _family(dataset_name: str) -> str | None:
    if dataset_name.startswith("drlDomain_"):
        return "drlDomain"
    if dataset_name.startswith("multisocial_"):
        return "multisocial"
    if dataset_name.startswith("tsm_"):
        return "tsm"
    if dataset_name.startswith("raid_"):
        return "raidModel"
    if dataset_name.startswith("raidModel_"):
        return "raidModel"
    return None


def _read_layer_aurocs(path: str) -> tuple[dict, np.ndarray] | None:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    args = obj.get("args")
    if not isinstance(args, dict):
        return None

    tm = obj.get("test_metrics", {})
    by_layer = tm.get("test_metrics_by_layer")
    if not isinstance(by_layer, list) or len(by_layer) == 0:
        return None

    vals = []
    for x in by_layer:
        if not isinstance(x, dict) or "auroc" not in x:
            return None
        vals.append(float(x["auroc"]))
    return args, np.asarray(vals, dtype=np.float64)


def collect_layer_curves(
    id_only: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], dict[str, list[str]]]:
    # family -> list of per-dimension AUROC vectors
    curves_by_family: dict[str, list[np.ndarray]] = defaultdict(list)
    files_by_family: dict[str, list[str]] = defaultdict(list)

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue

        read_out = _read_layer_aurocs(os.path.join(PROBE_DIR, filename))
        if read_out is None:
            continue

        args, layer_aurocs = read_out
        mode = args.get("mode")
        src = args.get("dataset")
        tgt = args.get("target_dataset")
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        if mode != MODE:
            continue

        fam = _family(src)
        if fam not in FAMILIES:
            continue
        if _family(tgt) != fam:
            continue

        if id_only and src != tgt:
            continue
        if (not id_only) and src == tgt:
            continue

        curves_by_family[fam].append(layer_aurocs)
        files_by_family[fam].append(filename)

    means: dict[str, np.ndarray] = {}
    cis: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    for fam in FAMILIES:
        arrs = curves_by_family.get(fam, [])
        if not arrs:
            continue

        n_layers = len(arrs[0])
        arrs = [a for a in arrs if len(a) == n_layers]
        if not arrs:
            continue

        stacked = np.stack(arrs, axis=0)  # (n_dims, n_layers)
        means[fam] = stacked.mean(axis=0)
        counts[fam] = stacked.shape[0]
        if stacked.shape[0] > 1:
            std = stacked.std(axis=0, ddof=1)
            sem = std / np.sqrt(stacked.shape[0])
            cis[fam] = 1.96 * sem
        else:
            cis[fam] = np.zeros_like(means[fam])

    return means, cis, counts, files_by_family


def _plot(
    means: dict[str, np.ndarray],
    cis: dict[str, np.ndarray],
    counts: dict[str, int],
    out_path: str,
    zoom: bool = False,
    vline_layer: int | None = None,
) -> None:
    if not means:
        raise RuntimeError("No ID default-mode per-layer AUROC curves found.")

    plt.figure(figsize=(10, 6))

    global_min = 1.0
    global_max = 0.0

    for fam in FAMILIES:
        if fam not in means:
            continue
        y = means[fam]
        ci = cis[fam]
        x = np.arange(len(y))
        global_min = min(global_min, float(np.min(y - ci)))
        global_max = max(global_max, float(np.max(y + ci)))

        low = np.clip(y - ci, 0.0, 1.0)
        high = np.clip(y + ci, 0.0, 1.0)
        plt.fill_between(x, low, high, color=COLORS[fam], alpha=0.18, linewidth=0)

        plt.plot(
            x,
            y,
            label=f"{FAMILY_LABELS.get(fam, fam)}",
            color=COLORS[fam],
            marker=MARKERS[fam],
            linewidth=2.0,
            markersize=5.5,
            alpha=0.95,
        )

    if vline_layer is not None:
        plt.axvline(vline_layer, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

    plt.xlabel("Layer", fontsize=FONT_SIZES["axis"])
    plt.ylabel("Mean AUROC", fontsize=FONT_SIZES["axis"])
    title_suffix = " (Zoomed)" if zoom else ""
    # plt.title(f"ID Per-Layer Mean AUROC by Dataset Family | mode={MODE}{title_suffix}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True, fontsize=FONT_SIZES["legend"])
    plt.xticks(fontsize=FONT_SIZES["ticks"])
    plt.yticks(fontsize=FONT_SIZES["ticks"])
    if zoom:
        low = max(0.0, global_min - 0.02)
        high = min(1.0, global_max + 0.02)
        if high - low < 0.05:
            mid = 0.5 * (high + low)
            low = max(0.0, mid - 0.025)
            high = min(1.0, mid + 0.025)
        plt.ylim(low, high)
    else:
        plt.ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for split_name, id_only in [("id", True), ("ood", False)]:
        means, cis, counts, files_by_family = collect_layer_curves(id_only=id_only)
        out_main = os.path.join(OUT_DIR, f"f_layer_{split_name}_default.pdf")
        vline_layer = 5 if split_name == "id" else 10
        _plot(means, cis, counts, out_main, zoom=True, vline_layer=vline_layer)

        print(f"Saved figure: {out_main}")
        print(f"{split_name.upper()} families plotted: {', '.join(sorted(means.keys()))}")
        max_expected = 4 if id_only else 12
        for fam in sorted(counts.keys()):
            print(f"{fam}: n_{split_name}_pairs={counts[fam]}")
            if counts[fam] > max_expected:
                print(
                    f"[Warning] {split_name.upper()} has more than expected files for family={fam}: "
                    f"{counts[fam]} > {max_expected}"
                )
                for name in files_by_family.get(fam, []):
                    print(f"  - {name}")


if __name__ == "__main__":
    main()
