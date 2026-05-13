import json
import os


BASE_DIR = os.getenv("BASE_COE", ".")
PROBE_DIR = os.path.join(BASE_DIR, "output", "probe", "sandbox")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")

DATASETS = [
    ("apt", "APT-Eval~\\cite{saha2025apt}"),
    ("editlens", "EditLens~\\cite{thai2026editlens}"),
]
MODE_ORDER = ["default", "meta_no_pca"]


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def _mode_label(mode: str) -> str:
    if mode == "default":
        return "LLP"
    if mode == "meta_no_pca":
        return "CLP"
    mode_latex = mode.replace("_", "\\_")
    return f"LP$_{{\\mathrm{{{mode_latex}}}}}$"


def _extract_corr_row(obj: dict) -> dict[str, float] | None:
    sim = obj.get("sim_correlations")
    if not isinstance(sim, dict) or not sim:
        return None

    # use first score entry (e.g., meta_scores / mean_projection / weighted_projection)
    first_key = next(iter(sim.keys()))
    row = sim.get(first_key)
    if not isinstance(row, dict):
        return None
    return row


def collect_rows() -> dict[str, dict[str, dict[str, float]]]:
    # dataset -> mode -> metric dict
    out: dict[str, dict[str, dict[str, float]]] = {"apt": {}, "editlens": {}}

    for filename in sorted(os.listdir(PROBE_DIR)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(PROBE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        args = obj.get("args", {})
        dataset = args.get("dataset")
        target_dataset = args.get("target_dataset")
        mode = args.get("mode")

        if dataset not in out:
            continue
        if dataset != target_dataset:
            continue
        if not mode:
            continue
        if mode not in {"default", "meta_no_pca"}:
            continue

        corr = _extract_corr_row(obj)
        if corr is None:
            continue

        out[dataset][mode] = corr

    return out


def render_table(rows: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & \\textbf{Semantic} $\\downarrow$ & \\textbf{Jaccard} $\\uparrow$ & \\textbf{Levenshtein} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    for dataset, panel_name in DATASETS:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{panel_name}}}}} \\\\")
        lines.append("\\midrule")

        dataset_rows = rows.get(dataset, {})
        ordered_modes = [m for m in MODE_ORDER if m in dataset_rows]
        ordered_modes.extend(m for m in dataset_rows if m not in ordered_modes)

        for mode in ordered_modes:
            corr = dataset_rows[mode]
            sim = _fmt(corr.get("sem_similarity"))
            jacca = _fmt(corr.get("jaccard_distance"))
            lev = _fmt(corr.get("levenshtein_distance"))
            lines.append(f"{_mode_label(mode)} & {sim} & {jacca} & {lev} \\\\")

        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = collect_rows()
    table = render_table(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "t_edits.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
