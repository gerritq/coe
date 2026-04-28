import json
import os
import re
from typing import Any


BASE_DIR = os.getenv("BASE_COE", ".")
BASELINE_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "output", "baseline", "sanbox"),
    os.path.join(BASE_DIR, "output", "baseline", "sandbox"),
]
PROBE_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "output", "probe", "sanbox"),
    os.path.join(BASE_DIR, "output", "probe", "sandbox"),
]
OUT_TEX = os.path.join(BASE_DIR, "item", "t_id.tex")


def _resolve_existing_dir(candidates: list[str]) -> str:
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"None of these directories exist: {candidates}")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _target_from_filename(filename: str) -> str | None:
    # Supports:
    # baseline: <model>_<source>_2_<target>.json
    # probe:    <token_mode>_<source>_2_<target>_pca{0,1}.json
    m = re.match(r"^.+_2_(.+)\.json$", filename)
    if not m:
        return None
    target = m.group(1)
    target = re.sub(r"_pca[01]$", "", target)
    return target


def _probe_row_name(args: dict[str, Any]) -> str:
    model = str(args.get("model", "probe"))
    token_mode = str(args.get("token_mode", "token"))
    pca = int(bool(args.get("pca", 0)))
    return f"{model}_{token_mode}_pca{pca}"


def _collect_baseline_scores(folder: str) -> dict[str, dict[str, float]]:
    # model -> (in-domain dataset -> auroc)
    scores: dict[str, dict[str, float]] = {}

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder, filename)
        payload = _read_json(path)
        args = payload.get("args", {})
        metrics = payload.get("metrics", {})

        source = args.get("dataset")
        model = args.get("model")
        target = _target_from_filename(filename)
        auroc = metrics.get("auroc", metrics.get("roc_auc"))
        if source is None or model is None or target is None or auroc is None:
            continue
        if str(source) != target:
            continue

        scores.setdefault(str(model), {})[str(source)] = float(auroc)

    return scores


def _collect_probe_scores(folder: str) -> dict[str, dict[str, float]]:
    # model -> (in-domain dataset -> auroc)
    scores: dict[str, dict[str, float]] = {}

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder, filename)
        payload = _read_json(path)
        args = payload.get("args", {})
        test_metrics = payload.get("test_metrics", {})
        ensemble_metrics = test_metrics.get("ensemble_metrics", {})

        source = args.get("dataset")
        target = _target_from_filename(filename)
        auroc = ensemble_metrics.get("auroc")
        if source is None or target is None or auroc is None:
            continue
        if str(source) != target:
            continue

        row_name = _probe_row_name(args)
        scores.setdefault(row_name, {})[str(source)] = float(auroc)

    return scores


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _format_auroc(value: float | None, best_value: float | None) -> str:
    if value is None:
        return ""
    formatted = f"{value:.3f}"
    if best_value is not None and abs(value - best_value) <= 1e-12:
        return rf"\textbf{{{formatted}}}"
    return formatted


def build_latex_table(
    baseline_scores: dict[str, dict[str, float]],
    probe_scores: dict[str, dict[str, float]],
) -> str:
    dataset_cols = sorted(
        set(ds for by_dataset in baseline_scores.values() for ds in by_dataset)
        | set(ds for by_dataset in probe_scores.values() for ds in by_dataset)
    )

    best_by_dataset: dict[str, float] = {}
    for ds in dataset_cols:
        vals: list[float] = []
        for by_dataset in baseline_scores.values():
            if ds in by_dataset:
                vals.append(by_dataset[ds])
        for by_dataset in probe_scores.values():
            if ds in by_dataset:
                vals.append(by_dataset[ds])
        if vals:
            best_by_dataset[ds] = max(vals)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l" + "c" * len(dataset_cols) + "}")
    lines.append(r"\hline")
    lines.append(
        "Model & "
        + " & ".join(_tex_escape(ds) for ds in dataset_cols)
        + r" \\"
    )
    lines.append(r"\hline")

    for model in sorted(baseline_scores):
        row = [_tex_escape(model)]
        for ds in dataset_cols:
            row.append(_format_auroc(baseline_scores[model].get(ds), best_by_dataset.get(ds)))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\hline")

    for model in sorted(probe_scores):
        row = [_tex_escape(model)]
        for ds in dataset_cols:
            row.append(_format_auroc(probe_scores[model].get(ds), best_by_dataset.get(ds)))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    baseline_dir = _resolve_existing_dir(BASELINE_DIR_CANDIDATES)
    probe_dir = _resolve_existing_dir(PROBE_DIR_CANDIDATES)

    baseline_scores = _collect_baseline_scores(baseline_dir)
    probe_scores = _collect_probe_scores(probe_dir)

    latex = build_latex_table(
        baseline_scores=baseline_scores,
        probe_scores=probe_scores,
    )

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {OUT_TEX}")


if __name__ == "__main__":
    main()
