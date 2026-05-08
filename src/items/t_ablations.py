import json
import os
import re

BASE_DIR = os.getenv("BASE_COE", ".")
ABLATION_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
DATASET_ORDER = ["tsm_first", "tsm_extend", "tsm_sums", "tsm_tst"]


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def _extract_auroc(obj: dict) -> float | None:
    mode = str(obj.get("args", {}).get("mode", ""))
    tm = obj.get("test_metrics", {})

    if "meta" in mode:
        meta = tm.get("meta_metrics", {})
        if "auroc" in meta:
            return float(meta["auroc"])

    mean = tm.get("mean_projection_metrics", {})
    if "auroc" in mean:
        return float(mean["auroc"])
    return None


def collect_pca_results() -> tuple[list[str], list[int], dict[int, dict[str, float]]]:
    # components -> dataset -> auroc
    table: dict[int, dict[str, float]] = {}
    datasets = set()

    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue
        if not filename.startswith("pca_"):
            continue

        path = os.path.join(ABLATION_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        if args.get("mode") != "pca":
            continue
        if args.get("token_mode") != "last_token":
            continue

        dataset = args.get("dataset")
        target = args.get("target_dataset")
        if not dataset or dataset != target:
            continue

        comp = args.get("components")
        try:
            comp = int(comp)
        except (TypeError, ValueError):
            # fallback parse from filename (e.g., PCA50)
            m = re.search(r"_PCA(\d+)_", filename)
            if not m:
                continue
            comp = int(m.group(1))

        auroc = _extract_auroc(obj)
        if auroc is None:
            continue

        datasets.add(dataset)
        table.setdefault(comp, {})[dataset] = auroc

    datasets_sorted = [d for d in DATASET_ORDER if d in datasets] + sorted(
        d for d in datasets if d not in DATASET_ORDER
    )
    comps_sorted = sorted(table.keys())
    return datasets_sorted, comps_sorted, table


def collect_default_results() -> dict[str, float]:
    table: dict[str, float] = {}
    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue
        if not filename.startswith("default_"):
            continue

        path = os.path.join(ABLATION_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        if args.get("mode") != "default":
            continue
        if args.get("token_mode") != "last_token":
            continue

        dataset = args.get("dataset")
        target = args.get("target_dataset")
        if not dataset or dataset != target:
            continue

        auroc = _extract_auroc(obj)
        if auroc is None:
            continue
        table[dataset] = auroc

    return table


def collect_default_by_token_results() -> dict[str, dict[str, float]]:
    # token_mode -> dataset -> auroc
    table: dict[str, dict[str, float]] = {}
    allowed = {"last_token", "pooling"}

    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue
        if not filename.startswith("default_"):
            continue

        path = os.path.join(ABLATION_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        if args.get("mode") != "default":
            continue

        token_mode = args.get("token_mode")
        if token_mode not in allowed:
            continue

        dataset = args.get("dataset")
        target = args.get("target_dataset")
        if not dataset or dataset != target:
            continue

        auroc = _extract_auroc(obj)
        if auroc is None:
            continue

        table.setdefault(token_mode, {})[dataset] = auroc

    return table


def collect_meta_results() -> tuple[list[int], dict[int, dict[str, float]], dict[str, float]]:
    # meta components -> dataset -> auroc, and meta_no_pca dataset -> auroc
    meta_table: dict[int, dict[str, float]] = {}
    meta_no_pca_table: dict[str, float] = {}

    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(ABLATION_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        mode = args.get("mode")
        if mode not in {"meta", "meta_no_pca"}:
            continue
        if args.get("token_mode") != "last_token":
            continue

        dataset = args.get("dataset")
        target = args.get("target_dataset")
        if not dataset or dataset != target:
            continue

        auroc = _extract_auroc(obj)
        if auroc is None:
            continue

        if mode == "meta_no_pca":
            meta_no_pca_table[dataset] = auroc
            continue

        comp = args.get("components")
        try:
            comp = int(comp)
        except (TypeError, ValueError):
            m = re.search(r"_PCA(\d+)_", filename)
            if not m:
                continue
            comp = int(m.group(1))
        meta_table.setdefault(comp, {})[dataset] = auroc

    return sorted(meta_table.keys()), meta_table, meta_no_pca_table


def collect_meta_attn_pca50_results() -> dict[str, float]:
    # dataset -> auroc for mode=meta_attn, token_mode=last_token, components=50, ID only
    table: dict[str, float] = {}

    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(ABLATION_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        if args.get("mode") != "meta_attn":
            continue
        if args.get("token_mode") != "last_token":
            continue

        dataset = args.get("dataset")
        target = args.get("target_dataset")
        if not dataset or dataset != target:
            continue

        comp = args.get("components")
        try:
            comp = int(comp)
        except (TypeError, ValueError):
            m = re.search(r"_PCA(\d+)_", filename)
            if not m:
                continue
            comp = int(m.group(1))
        if comp != 50:
            continue

        auroc = _extract_auroc(obj)
        if auroc is None:
            continue
        table[dataset] = auroc

    return table


def render_table(
    datasets: list[str],
    comps: list[int],
    values: dict[int, dict[str, float]],
    default_values: dict[str, float],
    default_by_token: dict[str, dict[str, float]],
    meta_comps: list[int],
    meta_values: dict[int, dict[str, float]],
    meta_no_pca_values: dict[str, float],
    meta_attn_pca50_values: dict[str, float],
) -> str:
    extra_datasets = set()
    for by_dataset in default_by_token.values():
        extra_datasets.update(by_dataset.keys())
    extra_datasets.update(meta_no_pca_values.keys())
    extra_datasets.update(meta_attn_pca50_values.keys())
    for by_dataset in meta_values.values():
        extra_datasets.update(by_dataset.keys())
    merged = set(datasets) | extra_datasets
    datasets = [d for d in DATASET_ORDER if d in merged] + sorted(d for d in merged if d not in DATASET_ORDER)

    cols = "l" + "c" * len(datasets)
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "\\multicolumn{%d}{l}{\\textbf{Panel A: PCA}} \\\\" % (len(datasets) + 1),
        "\\midrule",
        "\\textbf{Components} & " + " & ".join(datasets) + " \\\\",
        "0 & " + " & ".join(_fmt(default_values.get(ds)) for ds in datasets) + " \\\\",
        "\\midrule",
    ]

    for c in comps:
        row = [_fmt(values.get(c, {}).get(ds)) for ds in datasets]
        lines.append(f"{c} & " + " & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{%d}{l}{\\textbf{Panel B: Default Token Mode}} \\\\"
            % (len(datasets) + 1),
            "\\midrule",
            "Mean Pooling & "
            + " & ".join(_fmt(default_by_token.get("pooling", {}).get(ds)) for ds in datasets)
            + " \\\\",
            "Last Token & "
            + " & ".join(_fmt(default_by_token.get("last_token", {}).get(ds)) for ds in datasets)
            + " \\\\",
        ]
    )

    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{%d}{l}{\\textbf{Panel C: Meta}} \\\\"
            % (len(datasets) + 1),
            "\\midrule",
            "Meta No PCA & "
            + " & ".join(_fmt(meta_no_pca_values.get(ds)) for ds in datasets)
            + " \\\\",
            "\\midrule",
        ]
    )
    for c in meta_comps:
        row = [_fmt(meta_values.get(c, {}).get(ds)) for ds in datasets]
        lines.append(f"Meta PCA {c} & " + " & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{%d}{l}{\\textbf{Panel D: Meta No PCA vs Meta Attn PCA 50}} \\\\"
            % (len(datasets) + 1),
            "\\midrule",
            "Meta PCA 50 & "
            + " & ".join(_fmt(meta_values.get(50, {}).get(ds)) for ds in datasets)
            + " \\\\",
            "Meta No PCA & "
            + " & ".join(_fmt(meta_no_pca_values.get(ds)) for ds in datasets)
            + " \\\\",
            "Meta Attn PCA 50 & "
            + " & ".join(_fmt(meta_attn_pca50_values.get(ds)) for ds in datasets)
            + " \\\\",
        ]
    )

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    datasets, comps, values = collect_pca_results()
    default_values = collect_default_results()
    default_by_token = collect_default_by_token_results()
    meta_comps, meta_values, meta_no_pca_values = collect_meta_results()
    meta_attn_pca50_values = collect_meta_attn_pca50_results()
    table = render_table(
        datasets,
        comps,
        values,
        default_values,
        default_by_token,
        meta_comps,
        meta_values,
        meta_no_pca_values,
        meta_attn_pca50_values,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ablation.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
