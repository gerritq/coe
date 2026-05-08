import json
import os
import re

BASE_DIR = os.getenv("BASE_COE", ".")
ABLATION_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def _extract_auroc(obj: dict) -> float | None:
    tm = obj.get("test_metrics", {})
    weighted = tm.get("weighted_projection_metrics", {})
    if "auroc" in weighted:
        return float(weighted["auroc"])
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

    datasets_sorted = sorted(datasets)
    comps_sorted = sorted(table.keys())
    return datasets_sorted, comps_sorted, table


def render_table(datasets: list[str], comps: list[int], values: dict[int, dict[str, float]]) -> str:
    cols = "l" + "c" * len(datasets)
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "\\multicolumn{%d}{l}{\\textbf{Panel 1: PCA}} \\\\" % (len(datasets) + 1),
        "\\midrule",
        "\\textbf{Components} & " + " & ".join(datasets) + " \\\\",
        "\\midrule",
    ]

    for c in comps:
        row = [_fmt(values.get(c, {}).get(ds)) for ds in datasets]
        lines.append(f"{c} & " + " & ".join(row) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    datasets, comps, values = collect_pca_results()
    table = render_table(datasets, comps, values)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ablation.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
