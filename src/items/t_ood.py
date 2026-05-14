import json
import os
from collections import defaultdict


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
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
    "raid": [
        "raid_cohere_chat",
        "raid_gpt4",
        "raid_llama_chat",
        "raid_mistral_chat",
    ],
}

FAMILY_ORDER = ["drlDomain", "multisocial", "tsm", "raid"]
FAMILY_TITLE = {
    "drlDomain": r"\textbf{DetectRL~\citep{wu2024detectrl}}",
    "multisocial": r"\textbf{MultiSocial~\citep{macko2025multi}}",
    "tsm": r"\textbf{TSM~\citep{quaremba2026tsm}}",
    "raid": r"\textbf{RAID~\cite{dugan2024raid}}",
}
SUBSET_LABELS = {
    "drlDomain_arxiv": r"\textbf{ArXiv}",
    "drlDomain_writing_prompt": r"\textbf{Reddit}",
    "drlDomain_yelp_review": r"\textbf{Yelp}",
    "drlDomain_xsum": r"\textbf{News}",
    "multisocial_en": r"\textbf{en}",
    "multisocial_de": r"\textbf{de}",
    "multisocial_ru": r"\textbf{ru}",
    "multisocial_zh": r"\textbf{zh}",
    "tsm_first": r"\textbf{FP}",
    "tsm_extend": r"\textbf{PE}",
    "tsm_sums": r"\textbf{SUM}",
    "tsm_tst": r"\textbf{TST}",
    "raid_cohere_chat": r"\textbf{Cohere}",
    "raid_gpt4": r"\textbf{GPT4}",
    "raid_llama_chat": r"\textbf{Llama}",
    "raid_mistral_chat": r"\textbf{Mistral}",
}

DATASET_ALIASES = {
    "raidModel_cohere_chat": "raid_cohere_chat",
    "raidModel_gpt4": "raid_gpt4",
    "raidModel_llama_chat": "raid_llama_chat",
    "raidModel_mistral_chat": "raid_mistral_chat",
}

# Same set as f_ood.py
METHOD_SPECS = [
    ("TextFluoroscopy", {"kind": "baseline", "model": "text_fluoroscopy"}),
    ("BiScope", {"kind": "baseline", "model": "biscope"}),
    ("RepreGuard", {"kind": "baseline", "model": "repreguard"}),
    ("RoBERTa", {"kind": "baseline", "model": "encoder"}),
    ("LLP", {"kind": "probe", "mode": "default"}),
    ("CLP", {"kind": "probe", "mode": "meta_no_pca"}),
]


def _dataset_to_family() -> dict[str, str]:
    out = {}
    for fam, ds_list in DATASET_GROUPS.items():
        for ds in ds_list:
            out[ds] = fam
    return out


def _canonical_dataset_name(ds: str | None) -> str | None:
    if ds is None:
        return None
    return DATASET_ALIASES.get(ds, ds)


def _probe_auroc(test_metrics: dict) -> float | None:
    meta = test_metrics.get("meta_metrics", {})
    if "auroc" in meta:
        return float(meta["auroc"])
    mm = test_metrics.get("mean_projection_metrics", {})
    if "auroc" in mm:
        return float(mm["auroc"])
    return None


def _collect_method_entries(spec: dict) -> dict[str, dict[str, dict[str, float]]]:
    ds_to_family = _dataset_to_family()
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    if spec["kind"] == "probe":
        for filename in sorted(os.listdir(PROBE_DIR)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(PROBE_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            args = obj.get("args", {})
            if args.get("mode") != spec["mode"]:
                continue

            train_ds = _canonical_dataset_name(args.get("dataset"))
            test_ds = _canonical_dataset_name(args.get("target_dataset"))
            if train_ds is None or test_ds is None:
                continue

            fam_train = ds_to_family.get(train_ds)
            fam_test = ds_to_family.get(test_ds)
            if fam_train is None or fam_train != fam_test:
                continue

            auroc = _probe_auroc(obj.get("test_metrics", {}))
            if auroc is None:
                continue
            out[fam_train][train_ds][test_ds] = auroc
    else:
        for filename in sorted(os.listdir(BASELINE_DIR)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(BASELINE_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            args = obj.get("args", {})
            if args.get("model") != spec["model"]:
                continue

            train_ds = _canonical_dataset_name(args.get("dataset"))
            test_ds = _canonical_dataset_name(args.get("target_dataset"))
            if train_ds is None or test_ds is None:
                continue

            fam_train = ds_to_family.get(train_ds)
            fam_test = ds_to_family.get(test_ds)
            if fam_train is None or fam_train != fam_test:
                continue

            auroc = obj.get("metrics", {}).get("auroc")
            if auroc is None:
                continue
            auroc = float(auroc)
            if spec["model"] == "repreguard" and auroc > 1.0:
                auroc = auroc / 100.0
            out[fam_train][train_ds][test_ds] = auroc

    return out


def _mean_ood_per_target(family: str, fam_entries: dict[str, dict[str, float]]) -> dict[str, float | None]:
    targets = DATASET_GROUPS[family]
    out: dict[str, float | None] = {}
    for tgt in targets:
        vals = []
        for src in targets:
            if src == tgt:
                continue
            v = fam_entries.get(src, {}).get(tgt)
            if v is not None:
                vals.append(float(v))
        if len(vals) != 3:
            print(f"[Warning] Expected 3 OOD entries for family={family} target={tgt} but found {len(vals)}")
        out[tgt] = (sum(vals) / len(vals)) if vals else None
    return out


def _print_ood_coverage_mismatches(
    method_name: str,
    entries: dict[str, dict[str, dict[str, float]]],
) -> None:
    for family in FAMILY_ORDER:
        targets = DATASET_GROUPS[family]
        for target in targets:
            expected_sources = {src for src in targets if src != target}
            found_sources = {
                src
                for src in targets
                if src != target
                if entries.get(family, {}).get(src, {}).get(target) is not None
            }
            if found_sources != expected_sources:
                missing = sorted(expected_sources - found_sources)
                extra = sorted(found_sources - expected_sources)
                print(
                    f"[OOD mismatch] method={method_name} family={family} target={target} "
                    f"expected=3 found={len(found_sources)} missing={missing} extra={extra}"
                )


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.3f}"

def _fmt_styled(v: float | None, style: str | None) -> str:
    base = _fmt(v)
    if base == "":
        return ""
    if style == "best":
        return f"\\cellcolor{{cyan!25}}\\textbf{{{base}}}"
    if style == "second":
        return f"\\cellcolor{{orange!25}}\\underline{{{base}}}"
    return base


def render_table(rows: dict[str, dict[str, float | None]]) -> str:
    all_subsets = []
    for fam in FAMILY_ORDER:
        all_subsets.extend(DATASET_GROUPS[fam])

    cols = "l" + "c" * len(all_subsets)
    n_data_cols = len(all_subsets)

    drl_n = len(DATASET_GROUPS["drlDomain"])
    ms_n = len(DATASET_GROUPS["multisocial"])
    tsm_n = len(DATASET_GROUPS["tsm"])
    raid_n = len(DATASET_GROUPS["raid"])

    s1 = 2
    e1 = s1 + drl_n - 1
    s2 = e1 + 1
    e2 = s2 + ms_n - 1
    s3 = e2 + 1
    e3 = s3 + tsm_n - 1
    s4 = e3 + 1
    e4 = s4 + raid_n - 1

    # Per-column styles across all models (tie-aware on displayed precision).
    style_map: dict[str, dict[str, str | None]] = {subset: {} for subset in all_subsets}
    model_names = [name for name, _ in METHOD_SPECS]
    for subset in all_subsets:
        vals = []
        for model_name in model_names:
            v = rows.get(model_name, {}).get(subset)
            if v is None:
                continue
            vals.append((model_name, round(float(v), 3)))
        if not vals:
            continue
        unique_scores = sorted({v for _, v in vals}, reverse=True)
        best = unique_scores[0]
        second = unique_scores[1] if len(unique_scores) > 1 else None
        for model_name, v in vals:
            if v == best:
                style_map[subset][model_name] = "best"
            elif second is not None and v == second:
                style_map[subset][model_name] = "second"
            else:
                style_map[subset][model_name] = None

    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "& \\multicolumn{%d}{c}{%s} & \\multicolumn{%d}{c}{%s} & \\multicolumn{%d}{c}{%s} & \\multicolumn{%d}{c}{%s} \\\\"
        % (drl_n, FAMILY_TITLE["drlDomain"], ms_n, FAMILY_TITLE["multisocial"], tsm_n, FAMILY_TITLE["tsm"], raid_n, FAMILY_TITLE["raid"]),
        "\\cmidrule(lr){%d-%d}\\cmidrule(lr){%d-%d}\\cmidrule(lr){%d-%d}\\cmidrule(lr){%d-%d}"
        % (s1, e1, s2, e2, s3, e3, s4, e4),
        "\\textbf{Model $\\downarrow$ / OOD $\\rightarrow$} & " + " & ".join(SUBSET_LABELS[s] for s in all_subsets) + " \\\\",
        "\\midrule",
    ]

    baseline_models = ["TextFluoroscopy", "BiScope", "RepreGuard", "RoBERTa"]
    probe_models = ["LLP", "CLP"]

    def _delta_str(v: float | None, b: float | None) -> str:
        if v is None or b is None:
            return ""
        d = (v - b) * 100.0
        if d >= 0:
            return f"\\textcolor{{green!60!black}}{{+{abs(d):.2f}}}"
        return f"\\textcolor{{orange!85!black}}{{-{abs(d):.2f}}}"

    # Best baseline per column
    best_baseline: dict[str, float | None] = {}
    for subset in all_subsets:
        vals = [rows.get(m, {}).get(subset) for m in baseline_models]
        vals = [float(v) for v in vals if v is not None]
        best_baseline[subset] = max(vals) if vals else None

    for model_name in baseline_models:
        vals = [
            _fmt_styled(rows.get(model_name, {}).get(subset), style_map[subset].get(model_name))
            for subset in all_subsets
        ]
        lines.append(f"{model_name} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")

    for model_name in probe_models:
        vals = [
            _fmt_styled(rows.get(model_name, {}).get(subset), style_map[subset].get(model_name))
            for subset in all_subsets
        ]
        lines.append(f"{model_name} & " + " & ".join(vals) + " \\\\")
        delta_vals = [
            _delta_str(rows.get(model_name, {}).get(subset), best_baseline.get(subset))
            for subset in all_subsets
        ]
        lines.append(r"\hspace*{1em}$\Delta$ vs BL & " + " & ".join(delta_vals) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    # model -> subset -> mean_ood
    table_rows: dict[str, dict[str, float | None]] = {}

    for model_name, spec in METHOD_SPECS:
        entries = _collect_method_entries(spec)
        _print_ood_coverage_mismatches(model_name, entries)
        subset_scores: dict[str, float | None] = {}
        for fam in FAMILY_ORDER:
            means = _mean_ood_per_target(fam, entries.get(fam, {}))
            subset_scores.update(means)
        table_rows[model_name] = subset_scores

    table = render_table(table_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "t_ood.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
