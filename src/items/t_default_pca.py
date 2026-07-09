import json
import math
import os


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_PATH = os.path.join(BASE_DIR, "output", "item", "t_default_pca_comparison.tex")

DATASET_GROUPS = [
    (
        "Multisocial",
        ["multisocial_en", "multisocial_de", "multisocial_ru", "multisocial_zh"],
    ),
    (
        "DetectRL Domains",
        [
            "drlDomain_arxiv",
            "drlDomain_writing_prompt",
            "drlDomain_yelp_review",
            "drlDomain_xsum",
        ],
    ),
    (
        "RAID Models",
        [
            "raidModel_cohere_chat",
            "raidModel_gpt4",
            "raidModel_llama_chat",
            "raidModel_mistral_chat",
        ],
    ),
]

DATASET_LABELS = {
    "multisocial_en": "en",
    "multisocial_de": "de",
    "multisocial_ru": "ru",
    "multisocial_zh": "zh",
    "drlDomain_arxiv": "ArXiv",
    "drlDomain_writing_prompt": "Reddit",
    "drlDomain_yelp_review": "Yelp",
    "drlDomain_xsum": "News",
    "raidModel_cohere_chat": "Cohere",
    "raidModel_gpt4": "GPT-4",
    "raidModel_llama_chat": "Llama",
    "raidModel_mistral_chat": "Mistral",
}

DATASET_ALIASES = {
    "raidModel_cohere_chat": ["raidModel_cohere_chat", "raid_cohere_chat"],
    "raidModel_gpt4": ["raidModel_gpt4", "raid_gpt4"],
    "raidModel_llama_chat": ["raidModel_llama_chat", "raid_llama_chat"],
    "raidModel_mistral_chat": ["raidModel_mistral_chat", "raid_mistral_chat"],
}


def _all_datasets() -> list[str]:
    return [dataset for _, datasets in DATASET_GROUPS for dataset in datasets]


def _probe_auroc(obj: dict) -> float | None:
    auroc = obj.get("test_metrics", {}).get("mean_projection_metrics", {}).get("auroc")
    if auroc is None:
        return None
    auroc = float(auroc)
    if math.isnan(auroc):
        return None
    return auroc


def collect_probes() -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {"default": {}, "pca": {}}
    best_rank: dict[tuple[str, str], tuple[bool, str, str]] = {}

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        mode = args.get("mode")
        source_dataset = args.get("dataset")
        target_dataset = args.get("target_dataset")
        if mode not in scores or source_dataset != target_dataset:
            continue

        auroc = _probe_auroc(obj)
        if auroc is None:
            continue

        for dataset in _all_datasets():
            aliases = DATASET_ALIASES.get(dataset, [dataset])
            if source_dataset not in aliases:
                continue

            key = (mode, dataset)
            rank = (source_dataset == dataset, str(args.get("datetime") or ""), filename)
            if key not in best_rank or rank > best_rank[key]:
                best_rank[key] = rank
                scores[mode][dataset] = auroc

    return scores


def _fmt(score: float) -> str:
    return f"{score:.3f}"


def _table_row(label: str, values: list[str]) -> str:
    return label + " & " + " & ".join(values) + r" \\"


def render_table(scores: dict[str, dict[str, float]]) -> str:
    datasets = _all_datasets()
    n_cols = len(datasets)
    lines = [
        r"\begin{tabular}{l" + "c" * n_cols + "}",
        r"\toprule",
        (
            r" & \multicolumn{4}{c}{Multisocial}"
            r" & \multicolumn{4}{c}{DetectRL Domains}"
            r" & \multicolumn{4}{c}{RAID Models} \\"
        ),
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}",
        "Mode & " + " & ".join(DATASET_LABELS[dataset] for dataset in datasets) + r" \\",
        r"\midrule",
        _table_row("default", [_fmt(scores["default"][dataset]) for dataset in datasets]),
        _table_row("pca", [_fmt(scores["pca"][dataset]) for dataset in datasets]),
        r"\midrule",
        _table_row(
            r"default $-$ pca",
            [_fmt(scores["default"][dataset] - scores["pca"][dataset]) for dataset in datasets],
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    scores = collect_probes()
    missing = [
        f"{mode}:{dataset}"
        for mode in ["default", "pca"]
        for dataset in _all_datasets()
        if dataset not in scores[mode]
    ]
    if missing:
        raise ValueError(f"Missing probe scores for {', '.join(missing)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(render_table(scores) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
