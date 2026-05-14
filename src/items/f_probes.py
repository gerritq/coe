import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.getenv("BASE_COE", ".")
DESC_DIR = os.path.join(BASE_DIR, "output", "desc")
OUT_DIR = os.path.join(BASE_DIR, "output", "item")


def _short_name(dataset: str) -> str:
    label_map = {
        "drlDomain_arxiv": "ArXiv",
        "drlDomain_writing_prompt": "Reddit",
        "drlDomain_yelp_review": "Yelp",
        "drlDomain_xsum": "News",
        "multisocial_en": "en",
        "multisocial_de": "de",
        "multisocial_ru": "ru",
        "multisocial_zh": "zh",
        "tsm_first": "FP",
        "tsm_extend": "PE",
        "tsm_sums": "SUM",
        "tsm_tst": "TST",
        "raidModel_cohere_chat": "Cohere",
        "raidModel_gpt4": "GPT4",
        "raidModel_llama_chat": "Llama",
        "raidModel_mistral_chat": "Mistral",
    }
    if dataset in label_map:
        return label_map[dataset]
    if dataset.startswith("drlDomain_"):
        return dataset.replace("drlDomain_", "")
    if dataset.startswith("multisocial_"):
        return dataset.replace("multisocial_", "")
    if dataset.startswith("tsm_"):
        return dataset.replace("tsm_", "")
    if dataset.startswith("raidModel_"):
        return dataset.replace("raidModel_", "")
    return dataset


def _find_probe_file(mode: str) -> str:
    matches = sorted(
        [
            p
            for p in Path(DESC_DIR).glob(f"probe_vectors_{mode}_*.json")
            if p.is_file()
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No probe vector file found for mode={mode} in {DESC_DIR}")
    if len(matches) > 1:
        print(f"[Warning] Multiple probe files for mode={mode}. Using latest: {matches[-1].name}")
    return str(matches[-1])


def _load_vectors(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    datasets = obj.get("datasets", [])
    vectors = obj.get("vectors", {})
    if not isinstance(datasets, list) or not isinstance(vectors, dict):
        raise ValueError(f"Invalid probe vector file format: {path}")

    x = []
    kept_datasets = []
    for ds in datasets:
        row = vectors.get(ds, {})
        probe = row.get("probe")
        if probe is None:
            continue
        vec = np.asarray(probe, dtype=np.float64)
        if vec.ndim != 1:
            continue
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec = vec / nrm
        x.append(vec)
        kept_datasets.append(ds)

    if len(x) == 0:
        raise RuntimeError(f"No valid probe vectors found in {path}")
    return kept_datasets, np.stack(x, axis=0)


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    # rows are already normalized, but keep robust normalization
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    x = x / norms
    return x @ x.T


def _plot_heatmap(sim: np.ndarray, datasets: list[str], out_path: str, title: str) -> None:
    labels = [_short_name(d) for d in datasets]
    family_labels = ["DetectRL", "MultiSocial", "TSM", "RAID"]
    plt.figure(figsize=(9.2, 7.2))
    im = plt.imshow(sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)

    # Group separators every 4 datasets.
    n = len(labels)
    for boundary in [3.5, 7.5, 11.5]:
        if boundary < n:
            plt.axhline(boundary, color="#777777", linewidth=0.8, alpha=0.55)
            plt.axvline(boundary, color="#777777", linewidth=0.8, alpha=0.55)

    # Group labels on top and left when 16 datasets are present.
    if n == 16:
        centers = [1.5, 5.5, 9.5, 13.5]
        ax = plt.gca()
        for c, fam in zip(centers, family_labels):
            ax.text(c, -0.5, fam, ha="center", va="bottom", fontsize=10, fontweight="bold", clip_on=False)
            ax.text(-1.80, c, fam, ha="right", va="center", fontsize=10, fontweight="bold", rotation=90, clip_on=False)

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine Similarity")
    plt.tight_layout(rect=(0.05, 0.05, 0.995, 0.995))
    plt.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    for mode, out_name in [("default", "f_probes.pdf"), ("pca", "f_probes_pca.pdf"), ("pca_space", "f_probes_pca_space.pdf")]:
        try:
            path = _find_probe_file(mode)
        except FileNotFoundError:
            print(f"[Warning] Skipping mode={mode}; no probe file found.")
            continue
        datasets, x = _load_vectors(path)
        if len(datasets) != 16:
            print(f"[Warning] Expected 16 datasets for mode={mode}, found {len(datasets)}")
        sim = _cosine_matrix(x)
        out_path = os.path.join(OUT_DIR, out_name)
        _plot_heatmap(sim, datasets, out_path, title=f"Probe Cosine Similarity ({mode})")


if __name__ == "__main__":
    main()
