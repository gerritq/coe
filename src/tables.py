import json
import os
import re
from dataclasses import dataclass

# define the subfolder here
SUBFOLDER_COE = "test"
SUBFOLDER_BASELINE = "test"

# base dirs
BASE_DIR = os.getenv("BASE_COE")
SCORER_DIR = os.path.join(BASE_DIR, "scores", SUBFOLDER_COE)
BASELINE_DIR = os.path.join(BASE_DIR, "baselines", SUBFOLDER_BASELINE)
TABLE_DIR = os.path.join(BASE_DIR, "tables")

SCORER_ORDER = ["gmm", "logistic", "mlp"]
METRIC = "acc"

@dataclass(frozen=True)
class Config:
    model: str
    mode: str
    diff_vectors: bool
    prefix: bool
    normalize: bool

    def as_row(self) -> list[str]:
        return [
            self.model,
            self.mode,
            "1" if self.diff_vectors else "0",
            "1" if self.prefix else "0",
            "1" if self.normalize else "0",
        ]


def parse_scorer(filename: str) -> str | None:
    m = re.match(r"^score_(gmm|logistic|mlp)_", filename)
    return m.group(1) if m else None


def load_scores() -> tuple[dict[Config, dict[str, dict[str, float]]], list[str]]:
    data: dict[Config, dict[str, dict[str, float]]] = {}
    datasets = set()

    for filename in sorted(os.listdir(SCORER_DIR)):
        if not filename.endswith(".json"):
            continue
        scorer = parse_scorer(filename)
        if scorer is None:
            continue

        path = os.path.join(SCORER_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        args = payload.get("args", {})
        metrics = payload.get("metrics", {})

        dataset = args.get("dataset")
        if dataset is None:
            continue

        config = Config(
            model=str(args.get("model")),
            mode=str(args.get("mode")),
            diff_vectors=bool(args.get("diff_vectors")),
            prefix=bool(args.get("prefix")),
            normalize=bool(args.get("normalize")),
        )

        metric = metrics.get(METRIC)
        if metric is None:
            continue

        datasets.add(dataset)
        data.setdefault(config, {}).setdefault(dataset, {})[scorer] = float(metric)

    return data, sorted(datasets)


def format_table(
    data: dict[Config, dict[str, dict[str, float]]],
    datasets: list[str],
) -> None:
    configs = list(data.keys())

    row_values: dict[Config, list[float | None]] = {}
    means: dict[Config, float | None] = {}
    num_score_cols = len(datasets) * len(SCORER_ORDER) + len(SCORER_ORDER)

    for config in configs:
        values: list[float | None] = []
        acc = 0.0
        count = 0
        scorer_acc = {scorer: 0.0 for scorer in SCORER_ORDER}
        scorer_count = {scorer: 0 for scorer in SCORER_ORDER}
        for ds in datasets:
            for scorer in SCORER_ORDER:
                val = data.get(config, {}).get(ds, {}).get(scorer)
                values.append(val)
                if val is not None:
                    acc += val
                    count += 1
                    scorer_acc[scorer] += val
                    scorer_count[scorer] += 1
        mean_val = acc / count if count > 0 else None
        scorer_means = [
            (scorer_acc[scorer] / scorer_count[scorer])
            if scorer_count[scorer] > 0
            else None
            for scorer in SCORER_ORDER
        ]
        values.extend(scorer_means)
        row_values[config] = values
        means[config] = mean_val

    configs = sorted(
        configs,
        key=lambda c: (
            c.model,
            -(means[c] if means[c] is not None else -1.0),
            c.mode,
            c.diff_vectors,
            c.prefix,
            c.normalize,
        ),
    )

    col_spec = "lllll" + "".join(["|ccc" for _ in datasets]) + "|ccc"
    lines = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    header = ["Model", "Mode", "DV", "Pre", "Norm"]
    for ds in datasets:
        header.append(f"\\multicolumn{{3}}{{c}}{{{ds}}}")
    header.append("\\multicolumn{3}{c}{Mean}")
    lines.append(" & ".join(header) + " \\\\")

    subheader = ["", "", "", "", ""]
    for _ in datasets:
        subheader.extend(SCORER_ORDER)
    subheader.extend(SCORER_ORDER)
    lines.append(" & ".join(subheader) + " \\\\")
    lines.append("\\midrule")

    per_model_max: dict[str, list[float | None]] = {}
    for config in configs:
        model = config.model
        if model not in per_model_max:
            per_model_max[model] = [None] * (num_score_cols + 1)
        for idx, val in enumerate(row_values[config]):
            if val is None:
                continue
            current = per_model_max[model][idx]
            if current is None or val > current:
                per_model_max[model][idx] = val

    def format_cell(val: float | None, max_val: float | None) -> str:
        if val is None:
            return ""
        formatted = f"{val*100:.2f}"
        if max_val is not None and abs(val - max_val) <= 1e-12:
            return f"\\textbf{{{formatted}}}"
        return formatted

    for config in configs:
        row = config.as_row()
        model_max = per_model_max.get(config.model, [])
        for idx, val in enumerate(row_values[config]):
            max_val = model_max[idx] if idx < len(model_max) else None
            row.append(format_cell(val, max_val))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    table = "\n".join(lines)
    table = table.replace("_", "-")
    os.makedirs(TABLE_DIR, exist_ok=True)
    out_dir = os.path.join(TABLE_DIR, f"t1_{METRIC}.tex")
    with open(out_dir, "w", encoding="utf-8") as f:
        f.write(table)


def load_baselines() -> tuple[dict[str, dict[str, float]], list[str]]:
    data: dict[str, dict[str, float]] = {}
    datasets = set()

    for filename in sorted(os.listdir(BASELINE_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(BASELINE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        args = payload.get("args", {})
        metrics = payload.get("metrics", {})

        dataset = args.get("dataset")
        model = args.get("model")
        if dataset is None or model is None:
            continue

        metric = metrics.get(METRIC)
        if metric is None:
            continue

        datasets.add(dataset)
        data.setdefault(str(model), {})[dataset] = float(metric)

    return data, sorted(datasets)


def format_baseline_table(
    data: dict[str, dict[str, float]],
    datasets: list[str],
) -> None:
    models = sorted(data.keys())

    col_spec = "l" + "".join(["|c" for _ in datasets])
    lines = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    header = ["Model"] + datasets
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for model in models:
        row = [model]
        for ds in datasets:
            val = data.get(model, {}).get(ds)
            row.append("" if val is None else f"{val*100:.2f}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    table = "\n".join(lines)
    table = table.replace("_", "-")
    os.makedirs(TABLE_DIR, exist_ok=True)
    out_dir = os.path.join(TABLE_DIR, f"baseline_{METRIC}.tex")
    with open(out_dir, "w", encoding="utf-8") as f:
        f.write(table)

def main() -> None:
    data, datasets = load_scores()
    table = format_table(data, datasets)
    baseline_data, baseline_datasets = load_baselines()
    format_baseline_table(baseline_data, baseline_datasets)



if __name__ == "__main__":
    main()
