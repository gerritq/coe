import json
import os
import re


BASE_DIR = os.getenv("BASE_COE", ".")
BASELINE_DIR = os.path.join(BASE_DIR, "output", "baseline", "sandbox")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

BASELINE_MODELS = ["text_fluoroscopy", "biscope", "encoder"]
PROBE_MODE = "default"
DETECTORS = ["text_fluoroscopy", "biscope", "encoder", "default"]

FAMILIES = {
    "detectrl": [
        "detectrl_arxiv",
        "detectrl_writing_prompt",
        "detectrl_yelp_review",
        "detectrl_xsum",
        "detectrl_dummy",
    ],
    "multisocial": [
        "multisocial_en",
        "multisocial_de",
        "multisocial_ru",
        "multisocial_zh",
        "multisocial_pt",
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

DETECTOR_LABELS = {
    "text_fluoroscopy": "TextFluoroscopy",
    "biscope": "BiScope",
    "encoder": "Encoder",
    "default": r"LP$_{\mathrm{default}}$",
}

FAMILY_LABELS = {
    "detectrl": r"DetectRL~\citep{wu2024detectrl}",
    "multisocial": r"MultiSocial~\citep{macko2025multi}",
    "tsm": r"TSM-Bench~\citep{quaremba2026tsm}",
}

SUBSET_LABELS = {
    "detectrl_arxiv": "ArXiv",
    "detectrl_writing_prompt": "WritingPrompts",
    "detectrl_yelp_review": "Yelp",
    "detectrl_xsum": "XSum",
    "detectrl_dummy": "",
    "multisocial_en": "en",
    "multisocial_de": "de",
    "multisocial_ru": "ru",
    "multisocial_zh": "zh",
    "multisocial_pt": "pt",
    "tsm_paras_en": "P-en",
    "tsm_paras_pt": "P-pt",
    "tsm_paras_vi": "P-vi",
    "tsm_sums_en": "S-en",
    "tsm_sums_pt": "S-pt",
    "tsm_sums_vi": "S-vi",
}


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _short_subset_name(ds: str) -> str:
    return SUBSET_LABELS.get(ds, ds)


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.3f}"


def _extract_probe_auroc(obj: dict) -> float | None:
    tm = obj.get("test_metrics", {})
    if "weighted_projection_metrics" in tm:
        return tm["weighted_projection_metrics"].get("auroc")
    if "meta_metrics" in tm:
        return tm["meta_metrics"].get("auroc")
    if "mean_projection_metrics" in tm:
        return tm["mean_projection_metrics"].get("auroc")
    return None


def load_aurocs() -> dict[str, dict[str, dict[str, float]]]:
    # detector -> train_subset -> eval_subset -> auroc
    data: dict[str, dict[str, dict[str, float]]] = {d: {} for d in DETECTORS}

    # Baselines
    for fn in sorted(os.listdir(BASELINE_DIR)):
        if not fn.endswith(".json"):
            continue
        model = None
        payload = fn[:-5]  # remove .json
        for mname in BASELINE_MODELS:
            prefix = f"{mname}_"
            if payload.startswith(prefix):
                model = mname
                payload = payload[len(prefix):]
                break
        if model is None:
            continue
        if "_2_" not in payload:
            continue
        train_ds, eval_ds = payload.split("_2_", 1)
        path = os.path.join(BASELINE_DIR, fn)
        with open(path, "r") as f:
            obj = json.load(f)
        auroc = obj.get("metrics", {}).get("auroc")
        if auroc is None:
            continue
        data[model].setdefault(train_ds, {})[eval_ds] = float(auroc)

    # Probes default mode
    for fn in sorted(os.listdir(PROBE_DIR)):
        if not fn.endswith(".json"):
            continue
        m = re.match(rf"^{PROBE_MODE}_last_token_(.+)_2_(.+)\.json$", fn)
        if not m:
            continue
        train_ds, eval_ds = m.group(1), m.group(2)
        path = os.path.join(PROBE_DIR, fn)
        with open(path, "r") as f:
            obj = json.load(f)
        auroc = _extract_probe_auroc(obj)
        if auroc is None:
            continue
        data["default"].setdefault(train_ds, {})[eval_ds] = float(auroc)

    return data


def render_family_table(
    family: str,
    subsets: list[str],
    data: dict[str, dict[str, dict[str, float]]],
    include_detector_header: bool,
) -> str:
    n = len(subsets)
    total_cols = 1 + n * len(DETECTORS)
    colspec = "l" + "c" * (n * len(DETECTORS))

    lines = [
        rf"% {family.upper()}",
        r"\noindent",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        rf"\multicolumn{{{total_cols}}}{{c}}{{\textbf{{{FAMILY_LABELS.get(family, family)}}}}} \\",
        r"\midrule",
    ]
    if include_detector_header:
        lines.append(r"\textbf{Train} " + " & ".join([]))
        detector_headers = []
        for d in DETECTORS:
            detector_headers.append(r"\multicolumn{{{}}}{{c}}{{\textbf{{{}}}}}".format(n, DETECTOR_LABELS[d]))
        lines[-1] = r"\textbf{Train} " + " & " + " & ".join(detector_headers) + r" \\"

    # cmidrules under detector blocks (always before subset row)
    cm = []
    start = 2
    for _ in DETECTORS:
        end = start + n - 1
        cm.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cm))

    subset_headers = " & ".join(_tex_escape(_short_subset_name(s)) for s in subsets)
    lines.append(r"\textbf{Subset} & " + " & ".join([subset_headers] * len(DETECTORS)) + r" \\")
    lines.append(r"\midrule")

    for train_ds in subsets:
        if train_ds.endswith("_dummy"):
            continue
        row = [_tex_escape(_short_subset_name(train_ds))]
        for det in DETECTORS:
            for eval_ds in subsets:
                val = data.get(det, {}).get(train_ds, {}).get(eval_ds)
                row.append(_fmt(val))
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    data = load_aurocs()
    families = ["detectrl", "multisocial", "tsm"]
    tables_by_family = {}
    for fam in families:
        tables_by_family[fam] = render_family_table(
            fam, FAMILIES[fam], data, include_detector_header=True
        )

    # Backward-compatible combined file.
    out_text = "\n\\par\\medskip\n\n".join(tables_by_family[f] for f in families)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "t_ood.tex")
    with open(out_path, "w") as f:
        f.write(out_text)
    print(f"Saved: {out_path}")

    # Split files for independent inclusion in LaTeX.
    split_paths = {
        "detectrl": os.path.join(OUT_DIR, "t_ood_detectrl.tex"),
        "multisocial": os.path.join(OUT_DIR, "t_ood_multisocial.tex"),
        "tsm": os.path.join(OUT_DIR, "t_ood_tsm.tex"),
    }
    for fam, p in split_paths.items():
        with open(p, "w") as f:
            f.write(tables_by_family[fam] + "\n")
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
