git pull --rebase

# Hanqi Qs
- Check with her whether the length ratio makes sense.
- How to find optimal normalization? Empirically?
- Her thoughts on length feature?
- Other ideas of CoE features?
- I expected more from the trajectories?


# To do
- can we do pca on the full activations, and the track metrics?

- can we collect all activations at all layers for all samples and the apply pca?
    - there is this paper that collects all in one matrix: Residual Stream Analysis of Overfitting And Structural Disruptions
- implment other norm variants, eg relative to previous
- add ideas of latent thinking to the overleaf
- check why bin is not working; may need to create separate env for falcon
- denoise activations for coe
- test the ood idea with gaussian fitting; try denoising before
    - we can have one machine and then a mix of machine attacks + normal human - > then fit gaussian on machine and rest is ood

# Baselines
- Add NPR
- Understand why Binoculars fails

**Papers**
- Actually read and understand the two papers

**Zero-shot thresholds**
- Understand how detectgpt, binoculars estimate the threshold? Or criteria?
- Implement baselines: lrr, ppl, ... etc..., fastdetectgpt, binoculars

**Full dimesion volatility score**
- Implement the dimension-independent score, but think how to aggregate?

# Brain dump

## Horizontal

- KD or JS divergence
- PPL change 
- logit vector, but this is highly dimensional

## Zero-shot

- Understand how Binoculars and FastDetectGPT get the thresholds?
    - I guess it is using ROC and then Youden's J
    - Using these to find the optimal threshold on a held out data

- Simple assignment to distributions? And then simple log diff?

- How to create a unified score with mean and std info?

## CoE Features

- Average change between the first and last token
- Dimension independent metric as in OOD paper
    - Square dim differences?

## Formulation as OOD
- COE as OOD with mahambolis distance
    - formulate as a OOD task of human text similar to the neurips paper
    -  same motivation; bc OOD issues, we should especially test this in our case
    - we need a reference distributions
    - then we only need machine texts and not human texts
    - can show first with pca 

## Misc

- Can we only use a subset of hidden states? Early, middle, late?
- PCA on hidden states?
- Other normalization: relative to first or last vector? On the dimension of the vector?
- Prefix: can we add is this text machine or human generated?
    - Visually there is not much change except for Wikipedia, but we should check in more detail
- Can we use the uncertainty at the unembedding layer across tokens?

- Other metrics
    - CoE for attention heads
    - Abs diff as in the neurips paper
    - Length change -> ratio of length
    - how the difference vector changes (magnitude and angle)
        - what about the normalization for those? Norm by the total path length or first vector?

    - absolute angle
    - Total turning as in 3.2 of Rintoul
    - Variation along the straight line between first and last one - tortuosity
    - distance from the average thought
- Use CoE embeddings for a classifier
 - Can use the dimension independent volatility vector as in "Embedding trajectory"
- Use CoT in some way?

**Done**
- Can we do this for AFC?
    - Tried with counterfact data, did not really work, distributions are overlapping af. My suspicion is that this is because the facts are really short 

# Literature

## Neurips: https://proceedings.neurips.cc/paper_files/paper/2024/file/4b734e95f0788a030a69caa987516186-Paper-Conference.pdf

- Define two types of trajectory volatilties
    - 

## https://www.osti.gov/servlets/purl/1319636

- Paper mentioned in CoE about trajectories

## Binoculars

- Zero-shot = no training data are used from the source LLM

- Two groups of MGT detection: (1) trained models on a binary classification task or a linear classifier on top of learned features, (2) using statistical signatures that are characteristic of MGT, which require little to none training data.
    - **My note**: this group has looked at horizontal statistics/across-token statistics/output space statistics -> but not at vertical statistics like CoE

**Method**

- Motivation: ppl alone is insufficient, some human text have low ppl (e.g., "continue 123") and some AI text have high ppl vs (e.g., "poem about Ronaldo as a tennis player") 

- Intuition: signal of how surprising the next token prediction is of one model to another

- (1) log ppl of a string s
    - Intuition: estimate of how likely a string is to the model

- (2) cross-ppl defined as: - 1/L sum over tokens: ppl_model_1 x log_ppl_model_2
    - For each token, how surprised is model_1 by the token prediction of model_2
    - This compares the prob distribution of next-word tokens, not the actual next word token
    - If pplx is low, then cross-ppl will be near one as both models agree
    - If pplx is high, then cross-ppl wil also be close to one, as for both observer and performer, the token is surprising
    - This acts as a baseline/normalizer for text difficulty

- (3) Binoculars is the ratio of M_1 log ppl / M_1_M_2 cross ppl
    - Intuition: cross ppl normalizes; how surprised is M_1 relative to the baseline ppl of M_2
    - Intuition 2: some texts have higher ppl, bc of the prompt ("continue 123" vs "poem about Ronaldo as a tennis player"). 

- Calibration: 
# Story
- Idea:
    - CoE: coe discrepancies may happen when LLMs generate correct and incorrect responses
    - We hypothezise the same but for machine vs human texts, i.e., the model "thinking process" (=coe) differs between these types of text
    - Task 1: how to quantify the coe features
    - Task 2: how to use those features to predict whether a text is machine or human