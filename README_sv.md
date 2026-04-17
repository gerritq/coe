git pull --rebase
:wq


# Today

## LDA idea

- NEXT: more OOD settings + baselines
    - more challenging ood settings + run proper baselines including repguard
- can be written as any  function that linrearly discriminates, we use lda
- multi m4 seems to beat the baseline
- correct the eval function, terrible eval scores 
- can we do lda in low dim?
- comparison to bert can be also about sample efficiency!
- can also compare the linear boundary identified by lr with the sv
- investigate where probes are most accurate
- take the subset of probes with highest acc
- we can also use this spans for making tokens interpretable, like going over the full text?
- we can work on identifying a subset of best probes, and then make this an ensemble


- m_lda better acc than lda; but we need to adjust eval fucntion
    - try denoising

- for lit review
 - papers relying on interals: repguard, https://aclanthology.org/2024.emnlp-main.885.pdf, Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features
- check also this for probes: https://aclanthology.org/2025.findings-emnlp.880.pdf


## Ablation idea

- ablation leads to low recall
    - maybe we should ablate times a a parameter alpha
    - NEXT: try other OOD


- full ablation as in "No training wheels"
    - but maybe we need to do this at all layers and tokens
- i think in the current implementation we also ablate the test set? I wasn't expecting that

- with pca like 500 the denoise and default plots are identical, how?
- !!! PCA on all layers not layer-wise

- Maybe also reduce to the last 2 thrid tokens?

    We extracted residual stream activations at each layer, averaging across all token positions within each story, beginning with the 50th token (at which point the emotional content should be apparent). 
    
- COE or SV: Gaussian and anomaly detection, similar to the coe anomaly paper and the neurips OOD paper
    - rather than training, estimating the machine direction/subspace and then ood is the distance to it
- revoew of detection methods: check of S5 multiplies scores
    - Just majority vote: https://arxiv.org/pdf/2402.13671

- can we use this to compute baslines: https://github.com/kinit-sk/IMGTB
- bin with new venv?
- can we collect many steering vectors and then pca? like eg on random boostrap?
- compute SV similarity acros datasets and layers
- read this: https://arxiv.org/pdf/2512.24574#page=4.91
 - noise reduction as in huang manifold steering

- for acc and f1 do threshold tuning instead of youden js
- take the ideas from https://www.anthropic.com/research/emotion-concepts-function

- instead of projecting on the SV; can we measure if it points away from the machine cluster?
    - Something like: 3. How to implement "Away-ness" (The RBF Kernel idea)Instead of a dot product, you can use a Distance Metric.Step 1: Find the centroid of your Machine cluster in the manifold: $\mu_{m}$.Step 2: For a new sample $h$, calculate the distance: $d = \|h - \mu_{m}\|^2$.Step 3: If $d$ is small, it "belongs" to the machine cluster. If $d$ is large, it's "pointing away" into the human or OOD space.

    - How can we identify that subspace?

- Other idea: see how the AI safety or hallucination works identify spaces/steer activations. There could be something we could do
    - Paper: https://www.alignmentforum.org/posts/72vpkRRvoPHKi48fi/truth-is-universal-robust-detection-of-lies-in-llms-3
    - https://arxiv.org/pdf/2410.00153
    - https://aclanthology.org/2025.naacl-short.47.pdf
- try pca to activation differences vs difference in means

- add notes from todist

- change the backbone of the models to the og papers


No, they are **not the same**, though they often point in similar directions. Here is the concise breakdown:

* **Difference in Means:** Calculates the vector between the **average** of Group A and the **average** of Group B. It treats all samples equally and only cares about the "center" of each cloud.
* **PCA (on contrastive pairs):** Finds the axis of **maximum variance** among the differences. If some pairs have a much stronger "signal" than others, PCA will align itself with those more influential samples.

**The "Cleaning" Intuition:**
* **Means:** Captures the average shift ($Signal + Average Noise$).
* **PCA:** Captures the most consistent trend ($Signal$ while ignoring inconsistent $Noise$).

In a "clean" dataset, they are nearly identical. In a "noisy" or "complex" dataset, **PCA** is a better "cleaner" because it ignores dimensions where the data is just randomly scattered.

- add tsm, and update hf
- email preksha
- fix bin with env
- poster and order

- add detect rl as a third corpus

- split projection scores across 10 layers
- include new metrics, the one in the paper AUC positive? 
    - Also include f1 and acc from true labels using the optimal threshold
    - inlcude this also for the baselines + new data loading
- create mulit-language version of M4
    - implemend ood flag and list of domains

- check why manifold is not working

- What is the idea of the steering in a manifold? Projection on different directions does this work?
- Cross-lingual, cross-domain, and cross-generator vs bert baseline
    - classifier new loading and OO-x implementation; should define the oo-x in the sh file
    - implement oo-x in steering

- Descriptives folder
    - Can we compute the similarity of SVs across languages and domains?
    - Can we do low dim projections of similarities
    - Can we compute sim of SVs in high and low manifold?
    - Can this be 3 plots?

    - sim of SVs in manifold vs SVs in high dim?
        - if we do this across langs and domans, and sim is higher in low-dim, we have successfully denoised

# Brain dump

## OOD

### General strategy

- Tune a threshold for a metric of choice (acc, f1, false-negative rate) on the validation set, and use this threshold on the test set

- Idea for OOD: ceteris paribus for domains, languages, generators, and prompting

- Find a way to compare the SV across settings

**OO domain**

- M4: generator: chatgpt; language: English; cross domain (wikihow, wikipedia, reddit, arxiv, peerread)

## Average projection position?

- get the average projection position, instead of the last layer

## Density idea

4. Implementation Idea: "Manifold Density"Instead of a simple dot product, you can use Mahalanobis Distance on the manifold.Project all validation samples into the $k$-dim PCA space.Calculate the Mean and Covariance of the Machine cluster in that low-dim space.For a test sample, calculate its distance to that cluster.Why this is better than a steering vector: A steering vector assumes the clusters are "blobs." A manifold density approach accounts for the fact that the "Machine" cluster might be "long and skinny" in some directions and "short" in others.

## Universal steering vector?

- Can we test whether the steering vector hold across domains, languages, and generators?
    - Test by obtaining a SV on one data dimension, and then generalizing to new ones. Compare this to bert.

- Can we do this for language with perfect parallel data?
    - We get parallel data and check whether the SV is the same for en, de, etc.
    - get sv from non-parallel data
    - if denoising shifts the SV closer to the true SV in 1, the we have evidence it works

## Denoise steering vector

- Idea to reduce the inteference noise that comes with the steering vector
- Via manifold -> why does this not work at the moment?
- Can we achieve this with a more diverse dataset?
- higher auroc if the denoising works
- can we obtain many steering vectors, there is also the idea on how to obtain the best one in the paper?
- or can we average the sv over layers and then get a single one at the end?
- or can we use the val set to obtain the most robust one? Eg we can use pca on the collected activations>

3. "Denoising" via PCA (The "Common Signal" approach)Often, the first Principal Component (PC1) of your difference vectors is the "General Machine Style," while PC2 or PC3 are "Domain Noise."Collect many difference vectors $\mathbf{s}_i = x_{m,i} - x_{h,i}$ across many different topics.Run PCA on these difference vectors.The First Principal Component is your "Denoised Steering Vector." It represents the shared signal that exists across all domains.


## Lower-dim projection

- Can we not only project on the steering vector, but on a low-dim manifold? On each PC? Or in the low-dim space?
    - And then get the norm in the subspace for example
- Gemini suggestions that we only look at the human or machine manifold?

## Effectiveness of SV

- Can we fool the model with the SV we obtained? Like llr, fast-detect, ppl: can we drop their performance with interventions?

## Misc

- Can we reverse the label? So that we assume that there is a homogenous machine subspace, but the human one differs?
    - And then projecting how to any of the human activation subspaces

- Can we identify the SV, and the clean it? Or like zero out noise and use the SLM with cleaned activations/SV for classification?

-  Can we FT a model first, and does this improves the SV? But how to fine-tune?

- Check what has been done in word embeddings, it may translate to SVs