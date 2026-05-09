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
    "m4": [
        "m4_bloomz",
        "m4_cohere",
        "m4_dolly",
        "m4_gpt4",
    ],
}

DATASET_LABELS = {
    "drlDomain_arxiv": r"\textbf{ArXiv}",
    "drlDomain_writing_prompt": r"\textbf{Reddit}",
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
    "m4_bloomz": r"\textbf{Bloomz}",
    "m4_cohere": r"\textbf{Cohere}",
    "m4_dolly": r"\textbf{Dolly}",
    "m4_gpt4": r"\textbf{GPT4}",
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
    "editlens": "EditLens",
    "encoder": "RoBERTa",
}

ZERO_SHOT_MODELS = [
    "entropy",
    "likelihood",
    "llr",
    "rank",
    "binoculars",
    "fastdetectgpt",
    "gescore",
    "repreguard",
]

SUPERVISED_MODELS = [
    "openai_roberta",
    "radar",
    "editlens",
    "raidar",
    "biscope",
    "text_fluoroscopy",
    "encoder",
]

PROBE_MODE_ORDER = ["default", "meta_no_pca"]


def _tex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _fmt(score: float | None) -> str:
    if score is None:
        return ""
    return f"{score:.3f}"

def _fmt_probe(score: float | None, style: str | None) -> str:
    base = _fmt(score)
    if base == "":
        return ""
    if style == "best":
        return f"\\cellcolor{{cyan!25}}\\textbf{{{base}}}"
    if style == "second":
        return f"\\cellcolor{{orange!25}}\\underline{{{base}}}"
    return base


def _probe_auroc(test_metrics: dict, mode: str | None) -> float | None:
    if mode in {"default", "pca"}:
        metrics = test_metrics.get("mean_projection_metrics", {})
    else:
        metrics = test_metrics.get("meta_metrics", {})
    return metrics.get("auroc")

def _all_datasets() -> list[str]:
    datasets = []
    for group in ["drlDomain", "multisocial", "tsm", "m4"]:
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
        # RepreGuard outputs are stored on a 0-100 scale in some files.
        if model == "repreguard" and auroc > 1.0:
            auroc = auroc / 100.0
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
    multisocial_n = len(DATASET_GROUPS["multisocial"])
    tsm_n = len(DATASET_GROUPS["tsm"])
    m4_n = len(DATASET_GROUPS["m4"])

    start_drldomain = 2
    end_drldomain = start_drldomain + drldomain_n - 1
    start_multisocial = end_drldomain + 1
    end_multisocial = start_multisocial + multisocial_n - 1
    start_tsm = end_multisocial + 1
    end_tsm = start_tsm + tsm_n - 1
    start_m4 = end_tsm + 1
    end_m4 = start_m4 + m4_n - 1

    # Keep only selected probe rows.
    ordered_probe_rows = []
    for mode in PROBE_MODE_ORDER:
        mode_latex = mode.replace("_", r"\_")
        key = f"LP$_{{\\mathrm{{{mode_latex}}}}}$"
        if key in probe_rows:
            ordered_probe_rows.append(key)

    # Global per-column styles across all displayed rows (baseline + selected probes).
    # dataset -> row_id -> style in {"best", "second", None}
    # row_id is baseline model key (e.g., "rank") or probe label key (e.g., "LP$_{...}$").
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
            v_float = float(v)
            # Some baselines (e.g., RepreGuard) are stored on a 0-100 scale.
            # Normalize to 0-1 for fair rank highlighting only.
            if v_float > 1.0:
                v_float = v_float / 100.0
            vals.append((row_id, round(v_float, 3)))
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
        "& \\multicolumn{{{}}}{{c}}{{\\textbf{{DetectRL~\\citep{{wu2024detectrl}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{MultiSocial~\\citep{{macko2025multi}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{TSM~\\citep{{quaremba2026tsm}}}}}} & \\multicolumn{{{}}}{{c}}{{\\textbf{{M4GT~\\cite{{wang2024m4gt}}}}}} \\\\".format(
            drldomain_n, multisocial_n, tsm_n, m4_n
        ),
        "\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}\\cmidrule(lr){{{}-{}}}".format(
            start_drldomain, end_drldomain,
            start_multisocial, end_multisocial,
            start_tsm, end_tsm,
            start_m4, end_m4,
        ),
        "\\textbf{Model} & " + " & ".join(DATASET_LABELS[d] for d in datasets) + " \\\\",
        "\\midrule",
    ]

    lines.append("\\multicolumn{%d}{l}{\\textbf{Zero-shot}} \\\\" % (n_data_cols + 1))
    lines.append("\\midrule")
    for model in ZERO_SHOT_MODELS:
        if model not in baseline_rows:
            continue
        vals = [
            _fmt_probe(baseline_rows[model].get(d), style_map[d].get(model))
            for d in datasets
        ]
        model_label = MODEL_LABELS.get(model, model)
        lines.append(f"{_tex_escape(model_label)} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{\\textbf{Supervised}} \\\\" % (n_data_cols + 1))
    lines.append("\\midrule")
    for model in SUPERVISED_MODELS:
        if model not in baseline_rows:
            continue
        vals = [
            _fmt_probe(baseline_rows[model].get(d), style_map[d].get(model))
            for d in datasets
        ]
        model_label = MODEL_LABELS.get(model, model)
        lines.append(f"{_tex_escape(model_label)} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{\\textbf{Linear Probes}} \\\\" % (n_data_cols + 1))
    lines.append("\\midrule")
    for model in ordered_probe_rows:
        vals = [
            _fmt_probe(probe_rows[model].get(d), style_map[d].get(model))
            for d in datasets
        ]
        lines.append(f"{model} & " + " & ".join(vals) + " \\\\")

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
