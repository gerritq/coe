import json
import os


BASE_DIR = os.getenv("BASE_COE", ".")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_PATH = os.path.join(BASE_DIR, "output", "item", "t_ece.tex")

DATASET_ORDER = [
    "drlDomain_arxiv",
    "drlDomain_writing_prompt",
    "drlDomain_yelp_review",
    "drlDomain_xsum",
    "multisocial_en",
    "multisocial_de",
    "multisocial_ru",
    "multisocial_zh",
    "raid_cohere_chat",
    "raid_gpt4",
    "raid_llama_chat",
    "raid_mistral_chat",
    "tsm_first",
    "tsm_extend",
    "tsm_sums",
    "tsm_tst",
]

DATASET_LABELS = {
    "drlDomain_arxiv": r"\textbf{ArXiv}",
    "drlDomain_writing_prompt": r"\textbf{Reddit}",
    "drlDomain_yelp_review": r"\textbf{Yelp}",
    "drlDomain_xsum": r"\textbf{News}",
    "multisocial_de": r"\textbf{de}",
    "multisocial_en": r"\textbf{en}",
    "multisocial_ru": r"\textbf{ru}",
    "multisocial_zh": r"\textbf{zh}",
    "raid_cohere_chat": r"\textbf{Cohere}",
    "raid_gpt4": r"\textbf{GPT4}",
    "raid_llama_chat": r"\textbf{Llama}",
    "raid_mistral_chat": r"\textbf{Mistral}",
    "tsm_extend": r"\textbf{PE}",
    "tsm_first": r"\textbf{FP}",
    "tsm_sums": r"\textbf{SUM}",
    "tsm_tst": r"\textbf{TST}",
}

DATASET_ALIASES = {
    "raidModel_cohere_chat": "raid_cohere_chat",
    "raidModel_gpt4": "raid_gpt4",
    "raidModel_llama_chat": "raid_llama_chat",
    "raidModel_mistral_chat": "raid_mistral_chat",
}


def _canonical_dataset_name(ds: str | None) -> str | None:
    if ds is None:
        return None
    return DATASET_ALIASES.get(ds, ds)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _probe_ece(test_metrics: dict, mode: str | None) -> float | None:
    if mode == "pca":
        return test_metrics.get("mean_projection_metrics", {}).get("ece")
    if mode == "meta":
        return test_metrics.get("meta_metrics", {}).get("ece")
    return None


def collect_scores() -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {"LLP": {}, "CLP": {}, "RoBERTa": {}}

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        ds = _canonical_dataset_name(args.get("dataset"))
        target_ds = _canonical_dataset_name(args.get("target_dataset"))
        mode = args.get("mode")
        if ds != target_ds or ds not in DATASET_ORDER:
            continue

        row = {"pca": "LLP", "meta": "CLP"}.get(mode)
        if row is None:
            continue

        ece = _probe_ece(obj.get("test_metrics", {}), mode)
        if ece is not None:
            scores[row][ds] = float(ece)

    for filename in sorted(os.listdir(BASELINE_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(BASELINE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        ds = _canonical_dataset_name(args.get("dataset"))
        target_ds = _canonical_dataset_name(args.get("target_dataset"))
        if args.get("model") != "encoder":
            continue
        if ds != target_ds or ds not in DATASET_ORDER:
            continue

        ece = obj.get("metrics", {}).get("ece")
        if ece is not None:
            scores["RoBERTa"][ds] = float(ece)

    return scores


def render_table(scores: dict[str, dict[str, float]]) -> str:
    datasets = [
        dataset
        for dataset in DATASET_ORDER
        if any(dataset in row_scores for row_scores in scores.values())
    ]
    cols = "l" + "c" * len(datasets)
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "\\textbf{Model} & " + " & ".join(DATASET_LABELS[d] for d in datasets) + r" \\",
        "\\midrule",
    ]

    for row in ["LLP", "CLP", "RoBERTa"]:
        vals = [_fmt(scores[row].get(dataset)) for dataset in datasets]
        lines.append(f"{row} & " + " & ".join(vals) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    scores = collect_scores()
    if not any(scores.values()):
        raise ValueError("No ECE values found for probes or encoder.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(render_table(scores) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
