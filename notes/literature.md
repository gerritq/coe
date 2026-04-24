# Steering Vectors

- OG SV paper is Turner et al., 2024
- Model refusal with SVs was first shown in Arditi et al., 2024


# AI Text Detectors

- use learn to distance descriptions

## Zero-shot

### Logits-based

**Binoculars** [Hans et al. (2024)](https://arxiv.org/pdf/2401.12070)

**FastDetectGPT**[Bao et al. (2024)](https://arxiv.org/pdf/2310.05130)

**LRR/NPR** [Su et al. (2023)](https://arxiv.org/pdf/2306.05540)

**Lastde** [Xu et al (2025)](https://openreview.net/pdf?id=vo4AHjowKi)

### Rewrite-based

**RAIDAR** [Mao et al. (2024)](https://arxiv.org/pdf/2401.12970)

**DNA-GPT** [Yang et al. (2024)](https://openreview.net/pdf?id=Xlayxj2fWp)

**GECScore** [Wu et al. 2025](https://aclanthology.org/2025.coling-main.684.pdf)

### Activations

**Intrinsic Dimension** [Tulchinskii et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/7baa48bc166aa2013d78cbdc15010530-Paper-Conference.pdf)


**RepGuard** [Chen et al., (2025)]

Data: DetectRL

Method: They compute a concept (steering) vector at each layer using PCA on the differences between human and machine samples, and then project new instances onto this vector. This is done at the token level, and the results are aggregated by averaging.

---

## ML-based

## Logits-based

**BIScope** [Guo et al. 2024](https://papers.nips.cc/paper_files/paper/2024/file/bc808cf2d2444b0abcceca366b771389-Paper-Conference.pdf)

## Rewrite-based


### Activations

**Text Fluoroscopy**

Data: self created (3 domains, 3 models)

Method: Based on the observation that semantic features are insufficient for distinguishing between human and machine text, the authors argue that middle layers best reflect the compositional process of mapping individual words into coherent sentences. They identify these "optimal" layers by projecting hidden states into the vocabulary space (via the logit lens) and selecting those with the largest distributional difference compared to the first and last layers. Features from these identified layers are then used to train a binary classifier for detection.

Notes: Identification of the middle layer is zero-shot, the probe is trained. Ablations are poor, there should be a comparison between the most accurate probe across layers vs. their layer selection. 

### Misc

**Ghostbuster** [Verma et al. (2024)](https://aclanthology.org/2024.naacl-long.95.pdf)

Data: self-created (3 domains)

Method: 1) Pass human and machine samples through various LMs (including bigram and davinci), 2) obtain word probabilites to combine these via scalar and vector functions into a set of small features, 3) train a logistic classifier on these features.
