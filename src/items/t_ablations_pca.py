import json
import os
from typing import Any


BASE_DIR = os.getenv("BASE_COE", ".")
ABLATION_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
SANDBOX_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_PATH = os.path.join(BASE_DIR, "output", "item", "t_ablation_pca.tex")

DATASETS = ["tsm_first", "tsm_extend", "tsm_sums", "tsm_tst"]
DATASET_LABELS = {
    "tsm_first": r"\textbf{FP}",
    "tsm_extend": r"\textbf{PE}",
    "tsm_sums": r"\textbf{SUM}",
    "tsm_tst": r"\textbf{TST}",
}


def _load_rows(directory: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(directory):
        return rows

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
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
                "model": args.get("model"),
                "mode": args.get("mode"),
                "token_mode": args.get("token_mode"),
                "training_size": args.get("training_size"),
                "components": args.get("components"),
                "C": args.get("C", 1.0),  # older files do not have C
                "auroc": float(auroc),
                "test_metrics_by_layer": metrics.get("test_metrics_by_layer"),
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


def _best_match_row(rows: list[dict[str, Any]], dataset: str, cond: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
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
            candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda x: float(x.get("auroc", float("-inf"))))


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def _fmt_delta_only(x: float | None, ref: float | None) -> str:
    if x is None or ref is None:
        return ""
    delta = x - ref
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.3f}"


def _render(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> str:
    # discover C for pca(100) last_token full-data
    c_values = sorted(
        {
            float(r.get("C", 1.0))
            for r in rows
            if r.get("mode") == "pca"
            and int(r.get("components")) == 100
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
        r" & \multicolumn{4}{c}{\textbf{TSM~\citep{quaremba2026tsm}}} \\",
        r"\cmidrule(lr){2-5}",
        r"\textbf{Setting} & "
        + " & ".join(DATASET_LABELS[d] for d in DATASETS)
        + r" \\",
        r"\midrule",
    ]

    ref_cond = {
        "model": "llama_8b",
        "mode": "pca",
        "components": 100,
        "token_mode": "last_token",
        "C": 1.0,
        "training_size_is_none": True,
    }
    ref_rows = {d: _best_match_row(baseline_rows, d, ref_cond) for d in DATASETS}
    ref_vals = {d: _best_match(baseline_rows, d, ref_cond) for d in DATASETS}
    lines.append("Baseline" + " & " + " & ".join(_fmt(ref_vals[d]) for d in DATASETS) + r" \\")

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{Token Aggregation}} \\")
    pooling_vals = [_best_match(rows, d, {"mode": "pca", "components": 100, "token_mode": "pooling", "C": 1.0, "training_size_is_none": True}) for d in DATASETS]
    lines.append(r"\hspace*{1em}Pooling" + " & " + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(pooling_vals, DATASETS)) + r" \\")

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{Layer Selection}} \\")
    first_layer_vals = []
    for d in DATASETS:
        tmbl = ref_rows[d].get("test_metrics_by_layer") if ref_rows[d] else None
        if isinstance(tmbl, list) and len(tmbl) > 0:
            first_layer_vals.append(tmbl[0].get("auroc"))
        else:
            first_layer_vals.append(None)
    lines.append(
        r"\hspace*{1em}First layer"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(first_layer_vals, DATASETS))
        + r" \\"
    )
    last_layer_vals = []
    for d in DATASETS:
        tmbl = ref_rows[d].get("test_metrics_by_layer") if ref_rows[d] else None
        if isinstance(tmbl, list) and len(tmbl) > 0:
            last_layer_vals.append(tmbl[-1].get("auroc"))
        else:
            last_layer_vals.append(None)
    lines.append(
        r"\hspace*{1em}Last layer"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(last_layer_vals, DATASETS))
        + r" \\"
    )

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{Model}} \\")
    llama_3b_vals = [
            _best_match(
                rows,
                d,
                {"model": "llama_3b", "mode": "pca", "components": 100, "token_mode": "last_token", "training_size_is_none": True},
            )
            for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}Llama-3B"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(llama_3b_vals, DATASETS))
        + r" \\"
    )
    llama_1b_vals = [
            _best_match(
                rows,
                d,
                {"model": "llama_1b", "mode": "pca", "components": 100, "token_mode": "last_token", "training_size_is_none": True},
            )
            for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}Llama-1B"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(llama_1b_vals, DATASETS))
        + r" \\"
    )
    qwen_32b_vals = [
            _best_match(
                rows,
                d,
                {"model": "qwen_32b", "mode": "pca", "components": 100, "token_mode": "last_token", "training_size_is_none": True},
            )
            for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}Qwen-32B"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(qwen_32b_vals, DATASETS))
        + r" \\"
    )
    qwen_8b_vals = [
            _best_match(
                rows,
                d,
                {"model": "qwen_8b", "mode": "pca", "components": 100, "token_mode": "last_token", "training_size_is_none": True},
            )
            for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}Qwen-8B"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(qwen_8b_vals, DATASETS))
        + r" \\"
    )
    qwen_4b_vals = [
            _best_match(
                rows,
                d,
                {"model": "qwen_4b", "mode": "pca", "components": 100, "token_mode": "last_token", "training_size_is_none": True},
            )
            for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}Qwen-4B"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(qwen_4b_vals, DATASETS))
        + r" \\"
    )
    qwen_06b_vals = [
            _best_match(
                rows,
                d,
                {"model": "qwen_06b", "mode": "pca", "components": 100, "token_mode": "last_token", "training_size_is_none": True},
            )
            for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}Qwen-0.6B"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(qwen_06b_vals, DATASETS))
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
                {"mode": "pca", "components": 100, "token_mode": "last_token", "C": c, "training_size_is_none": True},
            )
            for d in DATASETS
        ]
        cells = [_fmt_delta_only(v, ref_vals[d]) for v, d in zip(vals, DATASETS)]
        lines.append(rf"\hspace*{{1em}}C={c:g}" + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{l}{\textbf{PCA Activations}} \\")
    keep_k = {10, 50, 150, 200, 250}
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

    # default (no pca) reference row in this panel
    no_pca_vals = [
        _best_match(
            rows,
            d,
            {"mode": "default", "components": 50, "token_mode": "last_token", "training_size_is_none": True},
        )
        for d in DATASETS
    ]
    lines.append(
        r"\hspace*{1em}No PCA"
        + " & "
        + " & ".join(_fmt_delta_only(v, ref_vals[d]) for v, d in zip(no_pca_vals, DATASETS))
        + r" \\"
    )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = _load_rows(ABLATION_DIR)
    baseline_rows = _load_rows(SANDBOX_DIR)
    table = _render(rows, baseline_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
