import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
QUAL_DIR = os.path.join(BASE_DIR, "output", "qual_metrics")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")
OUT_PATH = os.path.join(OUT_DIR, "f_qual_if.pdf")

DOMAIN_ORDER = [
    "d_m4_domains_wikipedia",
    "d_m4_domains_arxiv",
    "d_m4_domains_reddit",
    "d_m4_domains_peerread",
]

METRIC_ORDER = ["angle", "length", "magnitude", "curvature"]

FONT = {
    "title": 14,
    "axis": 13,
    "ticks": 11,
    "legend": 12,
}


def _metric_label(metric: str) -> str:
    labels = {
        "angle": "Angle",
        "length": "Length",
        "magnitude": "Magnitude",
        "curvature": "Curvature",
    }
    return labels.get(metric, metric)


def _domain_label(dataset: str) -> str:
    if dataset.startswith("d_m4_domains_"):
        return dataset.replace("d_m4_domains_", "")
    return dataset


def _collect_records() -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not os.path.isdir(QUAL_DIR):
        return grouped

    for filename in sorted(os.listdir(QUAL_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(QUAL_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        dataset = obj.get("dataset")
        metric = obj.get("metric")
        seed = obj.get("seed")
        layers = obj.get("layers")
        human = obj.get("human")
        machine = obj.get("machine")

        if dataset not in DOMAIN_ORDER:
            continue
        if metric not in METRIC_ORDER:
            continue
        if not isinstance(seed, int):
            continue
        if not isinstance(layers, list) or not isinstance(human, list) or not isinstance(machine, list):
            continue
        if len(layers) == 0 or len(layers) != len(human) or len(layers) != len(machine):
            continue

        grouped[(dataset, metric)].append(
            {
                "seed": seed,
                "layers": np.asarray(layers, dtype=np.int64),
                "human": np.asarray(human, dtype=np.float64),
                "machine": np.asarray(machine, dtype=np.float64),
                "filename": filename,
            }
        )
    return grouped


def _aggregate(grouped: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for key, records in sorted(grouped.items()):
        dataset, metric = key

        by_seed: dict[int, dict] = {}
        for rec in records:
            by_seed[rec["seed"]] = rec
        selected = [by_seed[s] for s in sorted(by_seed.keys())]

        seeds = sorted(by_seed.keys())
        if len(seeds) != 5:
            print(
                f"[Warning] Expected 5 seed files for dataset={dataset}, metric={metric} "
                f"but found {len(seeds)} seeds: {seeds}"
            )

        n_layers = len(selected[0]["layers"])
        selected = [r for r in selected if len(r["layers"]) == n_layers]
        if not selected:
            continue

        layers = selected[0]["layers"]
        h_stack = np.stack([r["human"] for r in selected], axis=0)
        m_stack = np.stack([r["machine"] for r in selected], axis=0)
        n = h_stack.shape[0]

        h_mean = h_stack.mean(axis=0)
        m_mean = m_stack.mean(axis=0)
        if n > 1:
            h_std = h_stack.std(axis=0, ddof=1)
            m_std = m_stack.std(axis=0, ddof=1)
            h_ci = 1.96 * h_std / np.sqrt(n)
            m_ci = 1.96 * m_std / np.sqrt(n)
        else:
            h_ci = np.zeros_like(h_mean)
            m_ci = np.zeros_like(m_mean)

        out[key] = {
            "layers": layers,
            "human_mean": h_mean,
            "machine_mean": m_mean,
            "human_ci": h_ci,
            "machine_ci": m_ci,
        }
    return out


def _plot(stats: dict[tuple[str, str], dict], out_path: str) -> None:
    if not stats:
        raise RuntimeError("No matching qual metrics found in output/qual_metrics.")

    fig, axes = plt.subplots(
        len(DOMAIN_ORDER),
        len(METRIC_ORDER),
        figsize=(4.5 * len(METRIC_ORDER), 3.1 * len(DOMAIN_ORDER)),
        squeeze=False,
    )

    for r, domain in enumerate(DOMAIN_ORDER):
        for c, metric in enumerate(METRIC_ORDER):
            ax = axes[r, c]
            key = (domain, metric)
            if key not in stats:
                ax.axis("off")
                continue

            st = stats[key]
            x = st["layers"]
            h_mean = st["human_mean"]
            m_mean = st["machine_mean"]
            h_ci = st["human_ci"]
            m_ci = st["machine_ci"]

            ax.plot(x, h_mean, color="#1f77b4", linewidth=1.6, marker="o", markersize=2.8, label="Human")
            ax.fill_between(x, h_mean - h_ci, h_mean + h_ci, color="#1f77b4", alpha=0.2, linewidth=0)

            ax.plot(x, m_mean, color="#d62728", linewidth=1.6, marker="o", markersize=2.8, label="Machine")
            ax.fill_between(x, m_mean - m_ci, m_mean + m_ci, color="#d62728", alpha=0.2, linewidth=0)

            if r == 0:
                ax.set_title(_metric_label(metric), fontsize=FONT["title"])
            if c == 0:
                ax.set_ylabel(_domain_label(domain), fontsize=FONT["axis"])
            else:
                ax.set_ylabel("")

            if r == len(DOMAIN_ORDER) - 1:
                ax.set_xlabel("Layer", fontsize=FONT["axis"])
            else:
                ax.set_xlabel("")

            ax.grid(alpha=0.25)
            ax.tick_params(axis="both", labelsize=FONT["ticks"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=True,
            fontsize=FONT["legend"],
            bbox_to_anchor=(0.5, -0.005),
        )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    grouped = _collect_records()
    stats = _aggregate(grouped)
    _plot(stats, out_path=OUT_PATH)


if __name__ == "__main__":
    main()
