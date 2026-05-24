import json
import os


BASE_DIR = os.getenv("BASE_COE", ".")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

ATTACK_DATASETS = [
    "drlAttack_multi_llm_mixing",
    "drlAttack_paraphrase_attacks_llm",
    "drlAttack_perturbation_attacks_llm",
    "drlAttack_prompt_attacks_llm",
]

DATASET_LABELS = {
    "drlAttack_multi_llm_mixing": r"\textbf{Mixing}",
    "drlAttack_paraphrase_attacks_llm": r"\textbf{Paraphrase}",
    "drlAttack_perturbation_attacks_llm": r"\textbf{Perturbation}",
    "drlAttack_prompt_attacks_llm": r"\textbf{Prompt}",
}

MODEL_LABELS = {
    "binoculars": "Binoculars",
    "biscope": "BiScope",
    "fastdetectgpt": "FastDetectGPT",
    "gescore": "GECScore",
    "likelihood": "Likelihood",
    "llr": "LLR",
    "openai_roberta": "OpenAI-RoBERTa",
    "radar": "RADAR",
    "raidar": "RAIDAR",
    "rank": "Rank",
    "repreguard": "RepreGuard",
    "text_fluoroscopy": "TextFluoroscopy",
    "editlens": "EditLens",
    "encoder": "RoBERTa",
    "revise": "Revise",
    "id": "ID",
}

ZERO_SHOT_MODELS = [
    "likelihood",
    "llr",
    "rank",
    "binoculars",
    "fastdetectgpt",
    "gescore",
    "revise",
    "raidar",
]

SUPERVISED_MODELS = [
    "openai_roberta",
    "radar",
    "editlens",
    "id",
    "repreguard",
    "biscope",
    "text_fluoroscopy",
    "encoder",
]

PROBE_MODE_ORDER = ["pca", "meta"]


def _tex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _fmt(score: float | None) -> str:
    if score is None:
        return ""
    return f"{score:.4f}"


def _fmt_probe(score: float | None, style: str | None) -> str:
    base = _fmt(score)
    if base == "":
        return ""
    if style == "best":
        return f"\\cellcolor{{cyan!25}}\\textbf{{{base}}}"
    if style == "second":
        return f"\\cellcolor{{orange!25}}\\underline{{{base}}}"
    return base


def _delta_str(v: float | None, b: float | None) -> str:
    if v is None or b is None:
        return ""
    d = (round(v, 4) - round(b, 4)) * 100.0
    if d >= 0:
        return f"\\textcolor{{green!60!black}}{{+{abs(d):.2f}}}"
    return f"\\textcolor{{orange!85!black}}{{-{abs(d):.2f}}}"


def _probe_auroc(test_metrics: dict, mode: str | None) -> float | None:
    if mode in {"default", "pca"}:
        metrics = test_metrics.get("mean_projection_metrics", {})
    else:
        metrics = test_metrics.get("meta_metrics", {})
    return metrics.get("auroc")


def collect_baselines() -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    datasets = set(ATTACK_DATASETS)

    for filename in sorted(os.listdir(BASELINE_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(BASELINE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        ds = args.get("dataset")
        target_ds = args.get("target_dataset")
        model = args.get("model")
        if ds != target_ds:
            continue
        if ds not in datasets or model is None:
            continue

        auroc = obj.get("metrics", {}).get("auroc")
        if auroc is None:
            continue
        auroc = float(auroc)
        if model == "repreguard" and auroc > 1.0:
            auroc = auroc / 100.0

        table.setdefault(model, {})[ds] = auroc

    return table


def collect_probes() -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    datasets = set(ATTACK_DATASETS)

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        ds = args.get("dataset")
        target_ds = args.get("target_dataset")
        mode = args.get("mode")
        components = args.get("components")

        if ds != target_ds:
            continue
        if ds not in datasets or mode is None:
            continue
        try:
            if int(components) != 100:
                continue
        except (TypeError, ValueError):
            continue

        if mode == "pca":
            row = "LLP"
        elif mode == "meta":
            row = "CLP"
        else:
            continue

        auroc = _probe_auroc(obj.get("test_metrics", {}), mode)
        if auroc is None:
            continue

        table.setdefault(row, {})[ds] = float(auroc)

    return table


def render_table(
    baseline_rows: dict[str, dict[str, float]],
    probe_rows: dict[str, dict[str, float]],
) -> str:
    datasets = ATTACK_DATASETS
    cols = "l" + "c" * len(datasets)

    ordered_probe_rows = []
    for mode in PROBE_MODE_ORDER:
        if mode == "pca":
            key = "LLP"
        elif mode == "meta":
            key = "CLP"
        else:
            continue
        if key in probe_rows:
            ordered_probe_rows.append(key)

    style_map: dict[str, dict[str, str | None]] = {d: {} for d in datasets}
    displayed_row_ids: list[str] = []
    displayed_row_ids.extend([m for m in ZERO_SHOT_MODELS if m in baseline_rows])
    displayed_row_ids.extend([m for m in SUPERVISED_MODELS if m in baseline_rows])
    displayed_row_ids.extend(ordered_probe_rows)

    for d in datasets:
        vals = []
        for row_id in displayed_row_ids:
            if row_id in baseline_rows:
                v = baseline_rows[row_id].get(d)
            else:
                v = probe_rows.get(row_id, {}).get(d)
            if v is None:
                continue
            vals.append((row_id, round(float(v), 4)))
        if not vals:
            continue
        unique_scores = sorted({v for _, v in vals}, reverse=True)
        best_score = unique_scores[0]
        second_score = unique_scores[1] if len(unique_scores) > 1 else None
        for row_id, v in vals:
            if v == best_score:
                style_map[d][row_id] = "best"
            elif second_score is not None and v == second_score:
                style_map[d][row_id] = "second"
            else:
                style_map[d][row_id] = None

    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "& \\multicolumn{4}{c}{\\textbf{DetectRL Attacks~\\citep{wu2024detectrl}}} \\\\",
        "\\cmidrule(lr){2-5}",
        "\\textbf{Model} & " + " & ".join(DATASET_LABELS[d] for d in datasets) + " \\\\",
        "\\midrule",
    ]

    lines.append("\\multicolumn{5}{l}{\\textbf{Zero-shot}} \\\\")
    lines.append("\\midrule")
    for model in ZERO_SHOT_MODELS:
        if model not in baseline_rows:
            continue
        vals = [_fmt_probe(baseline_rows[model].get(d), style_map[d].get(model)) for d in datasets]
        lines.append(f"{_tex_escape(MODEL_LABELS.get(model, model))} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\textbf{Supervised}} \\\\")
    lines.append("\\midrule")
    for model in SUPERVISED_MODELS:
        if model not in baseline_rows:
            continue
        vals = [_fmt_probe(baseline_rows[model].get(d), style_map[d].get(model)) for d in datasets]
        lines.append(f"{_tex_escape(MODEL_LABELS.get(model, model))} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\textbf{Linear Probes}} \\\\")
    lines.append("\\midrule")

    baseline_models = [m for m in (ZERO_SHOT_MODELS + SUPERVISED_MODELS) if m in baseline_rows]
    best_baseline: dict[str, float | None] = {}
    for d in datasets:
        vals = [baseline_rows[m].get(d) for m in baseline_models]
        vals = [float(v) for v in vals if v is not None]
        best_baseline[d] = max(vals) if vals else None

    for model in ordered_probe_rows:
        vals = [_fmt_probe(probe_rows[model].get(d), style_map[d].get(model)) for d in datasets]
        lines.append(f"{model} & " + " & ".join(vals) + " \\\\")
        delta_vals = [_delta_str(probe_rows[model].get(d), best_baseline.get(d)) for d in datasets]
        lines.append(r"\hspace*{1em}$\Delta$ vs BL & " + " & ".join(delta_vals) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    baselines = collect_baselines()
    probes = collect_probes()
    table = render_table(baselines, probes)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_main = os.path.join(OUT_DIR, "t_id_attacks_pca.tex")
    with open(out_main, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"Saved: {out_main}")


if __name__ == "__main__":
    main()
