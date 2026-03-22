from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass

BASE_DIR = os.getenv("BASE_COE")
SCORER_DIR = os.path.join(BASE_DIR, "scores")
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
) -> str:
    configs = sorted(
        data.keys(),
        key=lambda c: (c.model, c.mode, c.diff_vectors, c.prefix, c.normalize),
    )

    col_spec = "lllll" + "".join(["|ccc" for _ in datasets])
    lines = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    header = ["Model", "Mode", "DV", "Pre", "Norm"]
    for ds in datasets:
        header.append(f"\\multicolumn{{3}}{{c}}{{{ds}}}")
    lines.append(" & ".join(header) + " \\\\")

    subheader = ["", "", "", "", ""]
    for _ in datasets:
        subheader.extend(SCORER_ORDER)
    lines.append(" & ".join(subheader) + " \\\\")
    lines.append("\\midrule")

    for config in configs:
        row = config.as_row()
        for ds in datasets:
            for scorer in SCORER_ORDER:
                val = data.get(config, {}).get(ds, {}).get(scorer)
                row.append("" if val is None else f"{val*100:.2f}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    table = "\n".join(lines)
    table = table.replace("_", "-")
    out_dir = os.path.join(TABLE_DIR, f"t1_{METRIC}.tex")
    with open(out_dir, "w", encoding="utf-8") as f:
        f.write(table)
    

def main() -> None:
    data, datasets = load_scores()
    table = format_table(data, datasets)



if __name__ == "__main__":
    main()
