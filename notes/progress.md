
# Progress

**21.03.26**

- coe:
    - Added flag for norm normalization change / no norm => rm seems to lead to more separable blobs
    - prefix => for Wikipedia, it leads to differences; but not for other domains. Need to compare results of scoring model with and w/o prefix; but this is an empircial question at the end

    - difference between first and last
    - something dimension based (with squared differences)
    
- chain of logits
    - Implemented mean/std entropy across tokens + mean/std of diff entropy across tokens; done this on full and top-k tokens => it could work, need to interpret and analyse results further
    - Implemented TVD => does also work to some extend but needs closer inspection + check whether this is correcty implemented


**22.03.26**
- Added three scorers mlp, gmm, and logistic and ran them on the three feature vectors => results are strong compared to bert baseline on actual texts

**26.03.2026 - Meeting with Hanqi**
- Do descriptives later
- Overall, we need to show better performance gains over existing methods
    - Classifier: this means testing OOD, OO Generator, OO language transferability, text length
    - TSM: upload correct data + explanations
    - Bin check with easy data + implement acc metrics
    - Add TSM + multisocial + another and compare to our
- We need a harder task
    - The motivation should be to show that other methods perform worse on a given task
    - We can use, e.g., paraphrasing: see chat
    - We can also inlcude perturbations: paraphrasing, random noise (or I guess just take other datasets than M4, e.g., our own?)
- New method: top-k and rank as in the reducing overthinking paper
    - Listen to the audio what she means by rank
    - recalculate embedding based on top-k svd
    - We should test the optimal number of top-k, varying the number and checking for how much variance they account similar to the overthinking paper
- PCA visualizations: maybe we can see from there what method we should develop
- Review of AI detection tools
    - See also the latest ones in Zotero
    

**01.04.2026**
- added m4 multilingual 
- added tsm data and detectrl

**02.04.2026**
- added the projection in manifold idea
- added detectrl
- found that m4 shared task data has an ood test set; changed this to an id one
    - our approch was poor on OOD; need to investigate
- ran denoising idea for various pcs -> need to analyse
    - looks like there are no big effects, but maybe the OOD is better
- changed data loading to from jsonl instead of ds
- new eval method with more metrics + trehsold is computed based on val


**10.04.2026**
- Tried pooling for sv, does not work at all. Last token is much better
- Tried denoising and ldp, applying pca per layer and across layers,
    - Per layer performs better

**26.04.2026**

- refactored the probing folder and code; now has a basic linear probing implementation
- added detectrl and multisocial as ds. We likely need some more challenging in domain data at least
-

# Learning

- Blog about manifolds: https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/

## Metrics

Metrics for text detection:

- tpr@x_fpr (RepreGuard, Neurips review): does not need a threshold

- false-negative rate (Binoculars)/true-positive rate: need threshold

## OOD

- OOD for zero shot means threshold tuning on a val set, using this threshold on data that has not been used in the val set


# Misc
- After each attention/FF block it is written to the residual stream, not after one block (see nanogpt)

- hidden states capture all proper hidden states. First is the embedding, and last is before logits.

- Magnitude change: This is the Eucledian distance between two vectors (not change in its length)
    - This is the Eucledian distance between two vectors (not change in its length)


- L2 norm = Eucledian distance = straight line difference between two vectors

- L1 norm = Manhatten distance = distance along grid lines to move between two vectors

- https://aclanthology.org/2026.findings-eacl.41.pdf + https://aclanthology.org/2024.acl-long.828.pdf : says that difference in means and pca on the difference vectors is bascially the same