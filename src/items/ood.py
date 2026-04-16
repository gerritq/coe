import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(os.getenv("BASE_COE", "."))
SV_OOD_DIR = BASE_DIR / "output" / "sv_ood"
BASELINE_DIR = SV_OOD_DIR / "sandbox_default"
METHOD_DIR = SV_OOD_DIR / "sandbox_clean_topic_val"
OUT_TEX = BASE_DIR / "item" / "ood.tex"


@dataclass
class ScoreRow:
    model: str
    steering_domain: str
    eval_domain: str
    ablation_set: str
    avg_auroc: float
    last_layer_auroc: float


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _last_layer_auroc(metrics_per_layer: dict[str, Any]) -> float:
    layer_items: list[tuple[int, float]] = []
    for layer_name, metrics in metrics_per_layer.items():
        if not layer_name.startswith("layer_"):
            continue
        try:
            idx = int(layer_name.split("_")[1])
            auc = float(metrics["auroc"])
        except (ValueError, KeyError, TypeError):
            continue
        layer_items.append((idx, auc))

    if not layer_items:
        return float("nan")
    layer_items.sort(key=lambda x: x[0])
    return layer_items[-1][1]


def _collect_rows(folder: Path) -> dict[tuple[str, str, str, str], ScoreRow]:
    rows: dict[tuple[str, str, str, str], ScoreRow] = {}
    for path in sorted(folder.glob("psm_*.json")):
        data = _read_json(path)

        steering_domain = str(data.get("steering_domain", ""))
        eval_domain = str(data.get("eval_domain", ""))
        if not steering_domain or not eval_domain:
            continue
        if steering_domain == eval_domain:
            # Keep only OOD pairs.
            continue

        args = data.get("args", {})
        model = str(args.get("model", ""))
        ablation_set = str(args.get("ablation_set", "all"))

        avg_auroc = float(data.get("metrics_avg_projection", {}).get("auroc", float("nan")))
        last_auroc = _last_layer_auroc(data.get("metrics_per_layer", {}))

        key = (model, steering_domain, eval_domain, ablation_set)
        rows[key] = ScoreRow(
            model=model,
            steering_domain=steering_domain,
            eval_domain=eval_domain,
            ablation_set=ablation_set,
            avg_auroc=avg_auroc,
            last_layer_auroc=last_auroc,
        )
    return rows


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def build_latex_table() -> str:
    baseline = _collect_rows(BASELINE_DIR)
    method = _collect_rows(METHOD_DIR)

    common_keys = sorted(set(baseline.keys()) & set(method.keys()))

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\begin{tabular}{llrrrrrr}"
    )
    lines.append(r"\hline")
    lines.append(
        r"Train & OOD & Base Avg & Clean Avg & $\Delta$ Avg & Base Last & Clean Last & $\Delta$ Last \\"
    )
    lines.append(r"\hline")

    for key in common_keys:
        base_row = baseline[key]
        new_row = method[key]

        d_avg = new_row.avg_auroc - base_row.avg_auroc
        d_last = new_row.last_layer_auroc - base_row.last_layer_auroc

        train_label = _tex_escape(base_row.steering_domain)
        ood_label = _tex_escape(base_row.eval_domain)

        lines.append(
            f"{train_label} & {ood_label} & "
            f"{base_row.avg_auroc:.3f} & {new_row.avg_auroc:.3f} & {d_avg:+.3f} & "
            f"{base_row.last_layer_auroc:.3f} & {new_row.last_layer_auroc:.3f} & {d_last:+.3f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{OOD AUROC comparison between baseline (\texttt{sandbox\_default}) and \texttt{clean\_topic\_val} (\texttt{sandbox\_clean\_topic\_val}). Metrics shown: average projection AUROC and last-layer AUROC.}"
    )
    lines.append(r"\label{tab:ood_baseline_vs_clean_topic_val}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    latex = build_latex_table()
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table to {OUT_TEX}")


if __name__ == "__main__":
    main()
