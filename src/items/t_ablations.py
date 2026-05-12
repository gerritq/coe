import json
import os
from typing import Any


BASE_DIR = os.getenv("BASE_COE", ".")
ABLATION_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
OUT_PATH = os.path.join(BASE_DIR, "output", "item", "ablation.tex")

DATASETS = ["tsm_first", "tsm_extend", "tsm_sums", "tsm_tst"]
DATASET_LABELS = {
    "tsm_first": r"\textbf{First}",
    "tsm_extend": r"\textbf{Extend}",
    "tsm_sums": r"\textbf{Sums}",
    "tsm_tst": r"\textbf{TST}",
}


def _load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(ABLATION_DIR):
        return rows

    for filename in sorted(os.listdir(ABLATION_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(ABLATION_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        args = obj.get("args", {})
        dataset = args.get("dataset")
        target = args.get("target_dataset")
        if dataset != target:
            continue
        if dataset not in DATASETS:
            continue

        metrics = obj.get("test_metrics", {})
        mp = metrics.get("mean_projection_metrics", {})
        auroc = mp.get("auroc")
        if auroc is None:
            continue

        rows.append(
            {
                "dataset": dataset,
                "mode": args.get("mode"),
                "token_mode": args.get("token_mode"),
                "training_size": args.get("training_size"),
                "components": args.get("components"),
                "C": args.get("C", 1.0),  # older files do not have C
                "auroc": float(auroc),
            }
        )
    return rows


def _is_full_data(v: Any) -> bool:
    return v in (None, "None")


def _best_match(rows: list[dict[str, Any]], dataset: str, cond: dict[str, Any]) -> float | None:
    candidates = []
    for r in rows:
        if r["dataset"] != dataset:
            continue
        ok = True
        for k, v in cond.items():
            if k == "training_size_is_none":
                if not _is_full_data(r.get("training_size")):
                    ok = False
                    break
            elif k == "C":
                if float(r.get("C", 1.0)) != float(v):
                    ok = False
                    break
            else:
                if r.get(k) != v:
                    ok = False
                    break
        if ok:
            candidates.append(r["auroc"])
    if not candidates:
        return None
    return max(candidates)


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def _fmt_delta_only(x: float | None, ref: float | None) -> str:
    if x is None or ref is None:
        return ""
    delta = x - ref
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.3f}"


def _render(rows: list[dict[str, Any]]) -> str:
    # discover C for default last_token full-data
    c_values = sorted(
        {
            float(r.get("C", 1.0))
            for r in rows
            if r.get("mode") == "default"
            and r.get("token_mode") == "last_token"
            and _is_full_data(r.get("training_size"))
        }
    )
    if not c_values:
        c_values = [1.0]

    # discover pca components for pca last_token full-data
    pca_components = sorted(
        {
            int(r.get("components"))
            for r in rows
            if r.get("mode") == "pca"
            and r.get("token_mode") == "last_token"
            and _is_full_data(r.get("training_size"))
            and r.get("components") is not None
        }
    )

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Setting} & "
        + " & ".join(DATASET_LABELS[d] for d in DATASETS)
        + r" \\",
        r"\midrule",
    ]

    ref_cond = {"mode": "default", "token_mode": "last_token", "C": 1.0, "training_size_is_none": True}
    ref_vals = {d: _best_match(rows, d, ref_cond) for d in DATASETS}
    lines.append("Baseline" + " & " + " & ".join(_fmt(ref_vals[d]) for d in DATASETS) + r" \\")

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{Token Aggregation}} \\")
    pooling_vals = [_best_match(rows, d, {"mode": "default", "token_mode": "pooling", "C": 1.0, "training_size_is_none": True}) for d in DATASETS]
    lines.append(r"\hspace*{1em}Pooling" + " & " + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(pooling_vals, DATASETS)) + r" \\")

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{Probe}} \\")
    mlp_vals = [_best_match(rows, d, {"mode": "mlp", "training_size_is_none": True}) for d in DATASETS]
    lines.append(r"\hspace*{1em}MLP" + " & " + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(mlp_vals, DATASETS)) + r" \\")

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{Layer Selection}} \\")
    first_layer_vals = [
        _best_match(
            rows,
            d,
            {"mode": "first_layer", "token_mode": "last_token", "C": 1.0, "training_size_is_none": True},
        )
        for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}First layer"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(first_layer_vals, DATASETS))
        + r" \\"
    )
    last_layer_vals = [
        _best_match(
            rows,
            d,
            {"mode": "last_layer", "token_mode": "last_token", "C": 1.0, "training_size_is_none": True},
        )
        for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}Last layer"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(last_layer_vals, DATASETS))
        + r" \\"
    )

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{Regularization Penalty}} \\")
    for c in c_values:
        if float(c) == 1.0:
            continue
        vals = [
            _best_match(
                rows,
                d,
                {"mode": "default", "token_mode": "last_token", "C": c, "training_size_is_none": True},
            )
            for d in DATASETS
        ]
        cells = [_fmt_delta_only(v, ref_vals[d]) for v, d in zip(vals, DATASETS)]
        lines.append(rf"\hspace*{{1em}}C={c:g}" + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{PCA Activations}} \\")
    keep_k = {10, 50, 100}
    for k in pca_components:
        if k not in keep_k:
            continue
        vals = [
            _best_match(
                rows,
                d,
                {"mode": "pca", "token_mode": "last_token", "components": k, "training_size_is_none": True},
            )
            for d in DATASETS
        ]
        cells = [_fmt_delta_only(v, ref_vals[d]) for v, d in zip(vals, DATASETS)]
        lines.append(rf"\hspace*{{1em}}k={k}" + " & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = _load_rows()
    table = _render(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
