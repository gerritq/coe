import json
import os
from typing import Any

BASE_DIR = os.getenv("BASE_COE")
SUBFOLDER_BASELINE = "sandbox"
SUBFOLDER_SV = "sandbox"
SUBFOLDER = SUBFOLDER_BASELINE

BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", SUBFOLDER_BASELINE)
SV_DIR = os.path.join(BASE_DIR, "output", "steering", SUBFOLDER_SV)
TABLE_DIR = os.path.join(BASE_DIR, "output", "item")

METRIC = "auroc"
DATASETS = [
    "tsm_multi",
    "m4_multi",
    "drl_t1_perturbation",
    "drl_t1_paraphrase",
    "multisocial_full",
]


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def load_baseline_scores() -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}

    if not os.path.isdir(BASELINE_DIR):
        return scores

    for filename in sorted(os.listdir(BASELINE_DIR)):
        if not filename.endswith(".json"):
            continue

        payload = _read_json(os.path.join(BASELINE_DIR, filename))
        args = payload.get("args", {})
        metrics = payload.get("metrics", {})

        model = str(args.get("model", ""))
        dataset = str(args.get("dataset", ""))
        metric_value = metrics.get(METRIC)

        if not model or dataset not in DATASETS or metric_value is None:
            continue

        scores.setdefault(model, {})[dataset] = float(metric_value)

    return scores


def _last_layer_metric(payload: dict[str, Any]) -> float | None:
    per_layer = payload.get("metrics_per_layer", {})
    if not isinstance(per_layer, dict) or not per_layer:
        return None

    n_layers = payload.get("n_layers")
    if isinstance(n_layers, int):
        layer_key = f"layer_{n_layers}"
        layer_metrics = per_layer.get(layer_key, {})
        metric_value = layer_metrics.get(METRIC)
        if metric_value is not None:
            return float(metric_value)

    def _layer_idx(key: str) -> int:
        if not key.startswith("layer_"):
            return -1
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return -1

    last_key = max(per_layer.keys(), key=_layer_idx)
    metric_value = per_layer.get(last_key, {}).get(METRIC)
    if metric_value is None:
        return None
    return float(metric_value)


def load_sv_scores() -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}

    if not os.path.isdir(SV_DIR):
        return scores

    for filename in sorted(os.listdir(SV_DIR)):
        if not filename.endswith(".json"):
            continue
        if not filename.startswith("svp_scores_"):
            continue

        payload = _read_json(os.path.join(SV_DIR, filename))
        args = payload.get("args", {})

        model = str(args.get("model", ""))
        dataset = str(payload.get("eval_domain", ""))
        metric_value = _last_layer_metric(payload)

        if not model or dataset not in DATASETS or metric_value is None:
            continue

        current = scores.setdefault(model, {}).get(dataset)
        if current is None or metric_value > current:
            scores[model][dataset] = metric_value

    return scores


def _table_row(
    name: str,
    scores: dict[str, float],
    best_by_dataset: dict[str, float],
) -> str:
    cells = [_latex_escape(name)]
    for dataset in DATASETS:
        value = scores.get(dataset)
        formatted = _format_metric(value)
        best_value = best_by_dataset.get(dataset)
        if (
            value is not None
            and best_value is not None
            and abs(value - best_value) <= 1e-12
        ):
            formatted = f"\\textbf{{{formatted}}}"
        cells.append(formatted)
    return " & ".join(cells) + " \\\\" 


def build_latex_table(
    baseline_scores: dict[str, dict[str, float]],
    sv_scores: dict[str, dict[str, float]],
) -> str:
    col_spec = "l" + "c" * len(DATASETS)
    lines: list[str] = []

    best_by_dataset: dict[str, float] = {}
    all_rows = list(baseline_scores.values()) + list(sv_scores.values())
    for dataset in DATASETS:
        values = [
            row_scores[dataset]
            for row_scores in all_rows
            if dataset in row_scores and row_scores[dataset] is not None
        ]
        if values:
            best_by_dataset[dataset] = max(values)

    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    header = ["Model"] + [_latex_escape(ds) for ds in DATASETS]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    lines.append("\\multicolumn{" + str(len(DATASETS) + 1) + "}{l}{\\textbf{Baseline}} \\\\")
    for model in sorted(baseline_scores.keys()):
        lines.append(_table_row(model, baseline_scores[model], best_by_dataset))

    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(DATASETS) + 1) + "}{l}{\\textbf{SV}} \\\\")
    for model in sorted(sv_scores.keys()):
        lines.append(_table_row(model, sv_scores[model], best_by_dataset))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    baseline_scores = load_baseline_scores()
    sv_scores = load_sv_scores()

    table = build_latex_table(
        baseline_scores=baseline_scores,
        sv_scores=sv_scores,
    )

    os.makedirs(TABLE_DIR, exist_ok=True)
    out_path = os.path.join(TABLE_DIR, f"sv_id_{METRIC}_{SUBFOLDER}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(out_path)


if __name__ == "__main__":
    main()
