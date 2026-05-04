import json
import os


BASE_DIR = os.getenv("BASE_COE", ".")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

DATASET_GROUPS = {
    "detectrl": [
        "detectrl_arxiv",
        "detectrl_writing_prompt",
        "detectrl_yelp_review",
        "detectrl_xsum",
    ],
    "multisocial": [
        "multisocial_de",
        "multisocial_en",
        "multisocial_pt",
        "multisocial_ru",
        "multisocial_zh",
    ],
    "tsm": [
        "tsm_paras_en",
        "tsm_paras_pt",
        "tsm_paras_vi",
        "tsm_sums_en",
        "tsm_sums_pt",
        "tsm_sums_vi",
    ],
}

DATASET_LABELS = {
    "detectrl_arxiv": r"\textbf{ArXiv}",
    "detectrl_writing_prompt": r"\textbf{WritingPrompts}",
    "detectrl_yelp_review": r"\textbf{Yelp}",
    "detectrl_xsum": r"\textbf{XSum}",
    "multisocial_de": r"\textbf{de}",
    "multisocial_en": r"\textbf{en}",
    "multisocial_pt": r"\textbf{pt}",
    "multisocial_ru": r"\textbf{ru}",
    "multisocial_zh": r"\textbf{zh}",
    "tsm_paras_en": r"\textbf{P-en}",
    "tsm_paras_pt": r"\textbf{P-pt}",
    "tsm_paras_vi": r"\textbf{P-vi}",
    "tsm_sums_en": r"\textbf{S-en}",
    "tsm_sums_pt": r"\textbf{S-pt}",
    "tsm_sums_vi": r"\textbf{S-vi}",
}

MODEL_LABELS = {
    "binoculars": "Binoculars",
    "biscope": "BiScope",
    "entropy": "Entropy",
    "fastdetectgpt": "FastDetectGPT",
    "gescore": "GEScore",
    "likelihood": "Likelihood",
    "llr": "LLR",
    "openai_roberta": "OpenAI-RoBERTa",
    "radar": "RADAR",
    "raidar": "RAIDAR",
    "rank": "Rank",
    "repreguard": "RepreGuard",
    "text_fluoroscopy": "TextFluoroscopy",
}

ZERO_SHOT_MODELS = [
    "binoculars",
    "entropy",
    "fastdetectgpt",
    "likelihood",
    "llr",
    "rank",
    "repreguard",
    "gescore",
]

SUPERVISED_MODELS = [
    "biscope",
    "openai_roberta",
    "raidar",
    "radar",
    "text_fluoroscopy",
]

PROBE_MODE_ORDER = ["default", "pca", "meta", "meta_attn"]


def _tex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _fmt(score: float | None) -> str:
    if score is None:
        return ""
    return f"{score:.3f}"


def _probe_auroc(test_metrics: dict) -> float | None:
    if "meta_metrics" in test_metrics:
        return test_metrics["meta_metrics"].get("auroc")
    if "weighted_projection_metrics" in test_metrics:
        return test_metrics["weighted_projection_metrics"].get("auroc")
    if "mean_projection_metrics" in test_metrics:
        return test_metrics["mean_projection_metrics"].get("auroc")
    return None


def _all_datasets() -> list[str]:
    datasets = []
    for group in ["detectrl", "multisocial", "tsm"]:
        datasets.extend(DATASET_GROUPS[group])
    return datasets


def collect_baselines() -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    datasets = set(_all_datasets())
    for filename in sorted(os.listdir(BASELINE_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(BASELINE_DIR, filename)
        with open(path, "r") as f:
            obj = json.load(f)
        args = obj.get("args", {})
        ds = args.get("dataset")
        model = args.get("model")
        if ds not in datasets or model is None:
            continue
        auroc = obj.get("metrics", {}).get("auroc")
        if auroc is None:
            continue
        table.setdefault(model, {})[ds] = auroc
    return table


def collect_probes() -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    datasets = set(_all_datasets())
    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r") as f:
            obj = json.load(f)
        args = obj.get("args", {})
        ds = args.get("dataset")
        mode = args.get("mode")
        if ds not in datasets or mode is None:
            continue
        mode_latex = mode.replace("_", r"\_")
        row = f"LP$_{{\\mathrm{{{mode_latex}}}}}$"
        auroc = _probe_auroc(obj.get("test_metrics", {}))
        if auroc is None:
            continue
        table.setdefault(row, {})[ds] = auroc
    return table


def render_table(
    baseline_rows: dict[str, dict[str, float]],
    probe_rows: dict[str, dict[str, float]],
) -> str:
    datasets = _all_datasets()
    cols = "l" + "c" * len(datasets)
    n_data_cols = len(datasets)

    detectrl_n = len(DATASET_GROUPS["detectrl"])
    multisocial_n = len(DATASET_GROUPS["multisocial"])
    tsm_n = len(DATASET_GROUPS["tsm"])

    start_detectrl = 2
    end_detectrl = start_detectrl + detectrl_n - 1
    start_multisocial = end_detectrl + 1
    end_multisocial = start_multisocial + multisocial_n - 1
    start_tsm = end_multisocial + 1
    end_tsm = start_tsm + tsm_n - 1

    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "& \\multicolumn{{{}}}{{c}}{{\\textbf{{DetectRL~\\citep{{wu2024detectrl}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{MultiSocial~\\citep{{macko2025multi}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{TSM-Bench~\\citep{{quaremba2026tsm}}}}}} \\\\".format(
            detectrl_n, multisocial_n, tsm_n
        ),
        "\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}".format(
            start_detectrl, end_detectrl,
            start_multisocial, end_multisocial,
            start_tsm, end_tsm,
        ),
        "\\textbf{Model} & " + " & ".join(DATASET_LABELS[d] for d in datasets) + " \\\\",
        "\\midrule",
    ]

    lines.append("\\multicolumn{%d}{l}{\\textbf{Zero-shot}} \\\\" % (n_data_cols + 1))
    lines.append("\\midrule")
    for model in ZERO_SHOT_MODELS:
        if model not in baseline_rows:
            continue
        vals = [_fmt(baseline_rows[model].get(d)) for d in datasets]
        model_label = MODEL_LABELS.get(model, model)
        lines.append(f"{_tex_escape(model_label)} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{\\textbf{Supervised}} \\\\" % (n_data_cols + 1))
    lines.append("\\midrule")
    for model in SUPERVISED_MODELS:
        if model not in baseline_rows:
            continue
        vals = [_fmt(baseline_rows[model].get(d)) for d in datasets]
        model_label = MODEL_LABELS.get(model, model)
        lines.append(f"{_tex_escape(model_label)} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{\\textbf{Linear Probes}} \\\\" % (n_data_cols + 1))
    lines.append("\\midrule")
    ordered_probe_rows = []
    for mode in PROBE_MODE_ORDER:
        key = f"LP$_{{\\mathrm{{{mode.replace('_', r'\\_')}}}}}$"
        if key in probe_rows:
            ordered_probe_rows.append(key)
    for model in probe_rows:
        if model not in ordered_probe_rows:
            ordered_probe_rows.append(model)

    for idx, model in enumerate(ordered_probe_rows):
        vals = [_fmt(probe_rows[model].get(d)) for d in datasets]
        lines.append(f"{model} & " + " & ".join(vals) + " \\\\")
        if idx == 1:
            lines.append("\\cdashline{2-%d}" % (n_data_cols + 1))

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    baselines = collect_baselines()
    probes = collect_probes()
    table = render_table(baselines, probes)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_main = os.path.join(OUT_DIR, "t_id.tex")
    with open(out_main, "w") as f:
        f.write(table)

    print(f"Saved: {out_main}")

if __name__ == "__main__":
    main()
