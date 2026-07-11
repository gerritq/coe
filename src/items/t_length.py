import json
import math
import os
from collections import defaultdict


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "ablation")
BASELINE_ABLATION_DIR = os.path.join(BASE_DIR, "output", "baseline", "ablation")
OUT_PATH = os.path.join(BASE_DIR, "output", "item", "t_length.tex")

MODE_ORDER = ["default", "pca", "meta", "encoder", "meta_no_pca", "mlp", "first_layer", "last_layer"]


def _auroc(obj: dict) -> float | None:
    args = obj.get("args", {})
    mode = args.get("mode")
    test_metrics = obj.get("test_metrics", {})

    if mode in {"meta", "meta_no_pca", "meta_attn"}:
        value = test_metrics.get("meta_metrics", {}).get("auroc")
    else:
        value = test_metrics.get("mean_projection_metrics", {}).get("auroc")

    if value is None:
        return None

    value = float(value)
    if math.isnan(value):
        return None
    return value


def collect_scores() -> dict[str, dict[int, list[float]]]:
    scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        if "max_chars" not in args:
            continue

        max_chars = int(args["max_chars"])
        if max_chars == -1:
            continue

        mode = args.get("mode")
        auroc = _auroc(obj)
        if mode is None or auroc is None:
            continue

        scores[str(mode)][max_chars].append(auroc)

    if os.path.isdir(BASELINE_ABLATION_DIR):
        for filename in sorted(os.listdir(BASELINE_ABLATION_DIR)):
            if not filename.endswith(".json"):
                continue

            path = os.path.join(BASELINE_ABLATION_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            args = obj.get("args", {})
            if args.get("model") != "encoder":
                continue
            if "max_chars" not in args:
                continue

            max_chars = int(args["max_chars"])
            if max_chars == -1:
                continue

            auroc = obj.get("metrics", {}).get("auroc")
            if auroc is None:
                continue

            auroc = float(auroc)
            if math.isnan(auroc):
                continue

            scores["encoder"][max_chars].append(auroc)

    return scores


def _fmt(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.3f}"


def _ordered_modes(scores: dict[str, dict[int, list[float]]]) -> list[str]:
    modes = [mode for mode in MODE_ORDER if mode in scores]
    modes.extend(sorted(mode for mode in scores if mode not in MODE_ORDER))
    return modes


def render_table(scores: dict[str, dict[int, list[float]]]) -> str:
    lengths = sorted({length for by_length in scores.values() for length in by_length})
    cols = "l" + "c" * len(lengths)

    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "Metric & " + " & ".join(str(length) for length in lengths) + r" \\",
        "\\midrule",
    ]

    for mode in _ordered_modes(scores):
        vals = [_fmt(scores[mode].get(length, [])) for length in lengths]
        lines.append(f"AUC ({mode}) & " + " & ".join(vals) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    scores = collect_scores()
    if not scores:
        raise ValueError("No ablation probe files found with args.max_chars != -1")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(render_table(scores) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
