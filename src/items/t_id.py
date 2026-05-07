import json
import os


BASE_DIR = os.getenv("BASE_COE", ".")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

DATASET_GROUPS = {
    "drlDomain": [
        "drlDomain_arxiv",
        "drlDomain_writing_prompt",
        "drlDomain_yelp_review",
        "drlDomain_xsum",
    ],
    "drlAttack": [
        "drlAttack_multi_llm_mixing",
        "drlAttack_paraphrase_attacks_llm",
        "drlAttack_perturbation_attacks_llm",
        "drlAttack_prompt_attacks_llm",
    ],
    "multisocial": [
        "multisocial_en",
        "multisocial_de",
        "multisocial_ru",
        "multisocial_zh",
    ],
    "tsm": [
        "tsm_first",
        "tsm_extend",
        "tsm_sums",
        "tsm_tst",
    ],
}

DATASET_LABELS = {
    "detectrl_arxiv": r"\textbf{ArXiv}",
    "detectrl_writing_prompt": r"\textbf{WritingPrompts}",
    "detectrl_yelp_review": r"\textbf{Yelp}",
    "detectrl_xsum": r"\textbf{XSum}",
    "drlDomain_arxiv": r"\textbf{ArXiv}",
    "drlDomain_writing_prompt": r"\textbf{WritingPrompts}",
    "drlDomain_yelp_review": r"\textbf{Yelp}",
    "drlDomain_xsum": r"\textbf{XSum}",
    "drlAttack_multi_llm_mixing": r"\textbf{Mixing}",
    "drlAttack_paraphrase_attacks_llm": r"\textbf{Paraphrase}",
    "drlAttack_perturbation_attacks_llm": r"\textbf{Perturbation}",
    "drlAttack_prompt_attacks_llm": r"\textbf{Prompt}",
    "multisocial_de": r"\textbf{de}",
    "multisocial_en": r"\textbf{en}",
    "multisocial_ru": r"\textbf{ru}",
    "multisocial_zh": r"\textbf{zh}",
    "tsm_extend": r"\textbf{Extend}",
    "tsm_first": r"\textbf{First}",
    "tsm_sums": r"\textbf{Sums}",
    "tsm_tst": r"\textbf{TST}",
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



def _probe_auroc(test_metrics: dict, mode: str | None) -> float | None:
    if mode in {"default", "pca"}:
        metrics = test_metrics.get("mean_projection_metrics", {})
    else:
        metrics = test_metrics.get("meta_metrics", {})
    return metrics.get("auroc")

def _all_datasets() -> list[str]:
    datasets = []
    for group in ["drlDomain", "drlAttack", "multisocial", "tsm"]:
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
        target_ds = args.get("target_dataset")
        mode = args.get("mode")
        # ID condition: train/source dataset must equal test/target dataset.
        if ds != target_ds:
            continue
        if ds not in datasets or mode is None:
            continue
        mode_latex = mode.replace("_", r"\_")
        row = f"LP$_{{\\mathrm{{{mode_latex}}}}}$"
        auroc = _probe_auroc(obj.get("test_metrics", {}), mode)
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

    drldomain_n = len(DATASET_GROUPS["drlDomain"])
    drlattack_n = len(DATASET_GROUPS["drlAttack"])
    multisocial_n = len(DATASET_GROUPS["multisocial"])
    tsm_n = len(DATASET_GROUPS["tsm"])

    start_drldomain = 2
    end_drldomain = start_drldomain + drldomain_n - 1
    start_drlattack = end_drldomain + 1
    end_drlattack = start_drlattack + drlattack_n - 1
    start_multisocial = end_drlattack + 1
    end_multisocial = start_multisocial + multisocial_n - 1
    start_tsm = end_multisocial + 1
    end_tsm = start_tsm + tsm_n - 1

    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "& \\multicolumn{{{}}}{{c}}{{\\textbf{{DetectRL~\\citep{{wu2024detectrl}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{DRL-Attack~\\citep{{wu2024detectrl}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{MultiSocial~\\citep{{macko2025multi}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{TSM-Bench~\\citep{{quaremba2026tsm}}}}}} \\\\".format(
            drldomain_n, drlattack_n, multisocial_n, tsm_n
        ),
        "\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}".format(
            start_drldomain, end_drldomain,
            start_drlattack, end_drlattack,
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
        mode_latex = mode.replace("_", r"\_")
        key = f"LP$_{{\\mathrm{{{mode_latex}}}}}$"
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
