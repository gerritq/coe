import json
import os
from collections import defaultdict
from typing import Any

# Existing subfolders to use
BASELINE_FOLDER = "0326_results_n500_binoculars_low"
SCORE_FOLDER = "0326_with_single_scores"

BASE_DIR = os.getenv("BASE_COE")
BASELINE_DIR = os.path.join(BASE_DIR, "baselines", BASELINE_FOLDER)
SCORE_DIR = os.path.join(BASE_DIR, "scores", SCORE_FOLDER)
TABLE_DIR = os.path.join(BASE_DIR, "tables")
OUT_PATH = os.path.join(TABLE_DIR, "zero_auc_table.tex")

PREFERRED_DATASETS = [
    "arxiv_chatgpt",
    "wikihow_chatgpt",
    "wikipedia_chatgpt",
    "reddit_chatgpt",
]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_baseline_rows(folder: str) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    if not os.path.isdir(folder):
        return rows

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        args = payload.get("args", {})
        metrics = payload.get("metrics", {})
        model = args.get("model")
        dataset = args.get("dataset")
        auc = _safe_float(metrics.get("auc")) if isinstance(metrics, dict) else None

        if not isinstance(model, str) or not isinstance(dataset, str) or auc is None:
            continue
        rows.append((model, dataset, auc))

    return rows


def _load_score_rows(folder: str) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    if not os.path.isdir(folder):
        return rows

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        args = payload.get("args", {})
        metrics = payload.get("metrics", {})
        model = args.get("model")
        dataset = args.get("dataset")
        if not isinstance(model, str) or not isinstance(dataset, str) or not isinstance(metrics, dict):
            continue

        for metric_name, metric_payload in metrics.items():
            if not isinstance(metric_payload, dict):
                continue
            auc = _safe_float(metric_payload.get("auc"))
            if auc is None:
                continue
            rows.append((f"{model}:{metric_name}", dataset, auc))

    return rows


def _collect_datasets(
    baseline_rows: list[tuple[str, str, float]],
    score_rows: list[tuple[str, str, float]],
) -> list[str]:
    all_datasets = {dataset for _, dataset, _ in baseline_rows + score_rows}
    ordered = [dataset for dataset in PREFERRED_DATASETS if dataset in all_datasets]
    ordered.extend(sorted(dataset for dataset in all_datasets if dataset not in ordered))
    return ordered[:4]


def _build_model_dataset_map(rows: list[tuple[str, str, float]]) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = defaultdict(dict)
    for model, dataset, auc in rows:
        current = data[model].get(dataset)
        if current is None or auc > current:
            data[model][dataset] = auc
    return dict(data)


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _format_cell(value: float | None, best: float | None) -> str:
    if value is None:
        return ""
    txt = f"{value:.3f}"
    if best is not None and abs(value - best) <= 1e-12:
        return rf"\textbf{{{txt}}}"
    return txt


def build_table() -> str:
    baseline_rows = _load_baseline_rows(BASELINE_DIR)
    score_rows = _load_score_rows(SCORE_DIR)

    datasets = _collect_datasets(baseline_rows, score_rows)
    baseline_map = _build_model_dataset_map(baseline_rows)
    score_map = _build_model_dataset_map(score_rows)

    baseline_models = sorted(baseline_map.keys())
    score_models = sorted(score_map.keys())

    best_per_dataset: dict[str, float | None] = {}
    for dataset in datasets:
        values: list[float] = []
        for model in baseline_models:
            val = baseline_map.get(model, {}).get(dataset)
            if val is not None:
                values.append(val)
        for model in score_models:
            val = score_map.get(model, {}).get(dataset)
            if val is not None:
                values.append(val)
        best_per_dataset[dataset] = max(values) if values else None

    col_spec = "l" + "c" * len(datasets)
    lines: list[str] = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Model & " + " & ".join(_latex_escape(ds) for ds in datasets) + r" \\",
        r"\midrule",
    ]

    for model in baseline_models:
        row = [_latex_escape(model)]
        for dataset in datasets:
            val = baseline_map.get(model, {}).get(dataset)
            row.append(_format_cell(val, best_per_dataset[dataset]))
        lines.append(" & ".join(row) + r" \\")

    if baseline_models and score_models:
        lines.append(r"\midrule")

    for model in score_models:
        display_model = model if model not in baseline_map else f"{model} (scores)"
        row = [_latex_escape(display_model)]
        for dataset in datasets:
            val = score_map.get(model, {}).get(dataset)
            row.append(_format_cell(val, best_per_dataset[dataset]))
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    table = build_table()
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
