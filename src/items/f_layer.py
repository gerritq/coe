import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

MODE = "default"
FAMILIES = ["detectrl", "multisocial", "tsm", "CB"]
COLORS = {
    "detectrl": "#1f77b4",
    "multisocial": "#d62728",
    "tsm": "#2ca02c",
    "CB": "#ff7f0e",
}
MARKERS = {
    "detectrl": "o",
    "multisocial": "s",
    "tsm": "^",
    "CB": "D",
}


def _family(dataset_name: str) -> str | None:
    if dataset_name.startswith("detectrl_"):
        return "detectrl"
    if dataset_name.startswith("multisocial_"):
        return "multisocial"
    if dataset_name.startswith("tsm_"):
        return "tsm"
    if dataset_name.startswith("CB_"):
        return "CB"
    return None


def _parse_filename(filename: str) -> tuple[str, str, str] | None:
    m = re.match(r"^(.*?)_last_token_(.+)_2_(.+)\.json$", filename)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _read_layer_aurocs(path: str) -> np.ndarray | None:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    tm = obj.get("test_metrics", {})
    by_layer = tm.get("test_metrics_by_layer")
    if not isinstance(by_layer, list) or len(by_layer) == 0:
        return None

    vals = []
    for x in by_layer:
        if not isinstance(x, dict) or "auroc" not in x:
            return None
        vals.append(float(x["auroc"]))
    return np.asarray(vals, dtype=np.float64)


def collect_layer_curves(id_only: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    # family -> list of per-dimension AUROC vectors
    curves_by_family: dict[str, list[np.ndarray]] = defaultdict(list)

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue

        parsed = _parse_filename(filename)
        if parsed is None:
            continue

        mode, src, tgt = parsed
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

        layer_aurocs = _read_layer_aurocs(os.path.join(PROBE_DIR, filename))
        if layer_aurocs is None:
            continue

        curves_by_family[fam].append(layer_aurocs)

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

    return means, cis, counts


def _plot(
    means: dict[str, np.ndarray],
    cis: dict[str, np.ndarray],
    counts: dict[str, int],
    out_path: str,
    zoom: bool = False,
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
            label=f"{fam} (n={counts[fam]})",
            color=COLORS[fam],
            marker=MARKERS[fam],
            linewidth=2.0,
            markersize=5.5,
            alpha=0.95,
        )

    plt.xlabel("Layer")
    plt.ylabel("Mean AUROC")
    title_suffix = " (Zoomed)" if zoom else ""
    plt.title(f"ID Per-Layer Mean AUROC by Dataset Family | mode={MODE}{title_suffix}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
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
        means, cis, counts = collect_layer_curves(id_only=id_only)
        out_main = os.path.join(OUT_DIR, f"{split_name}_probe_layer_auroc_default.pdf")
        out_zoom = os.path.join(OUT_DIR, f"{split_name}_probe_layer_auroc_default_zoom.pdf")

        _plot(means, cis, counts, out_main, zoom=False)
        _plot(means, cis, counts, out_zoom, zoom=True)

        print(f"Saved figure: {out_main}")
        print(f"Saved figure: {out_zoom}")
        print(f"{split_name.upper()} families plotted: {', '.join(sorted(means.keys()))}")
        for fam in sorted(counts.keys()):
            print(f"{fam}: n_{split_name}_pairs={counts[fam]}")


if __name__ == "__main__":
    main()
