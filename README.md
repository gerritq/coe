git pull --rebase

# Hanqi Qs
- Check with her whether the length ratio makes sense.
- How to find optimal normalization? Empirically?
- Her thoughts on length feature?
- Other ideas of CoE features?
- I expected more from the trajectories?


# To do

- add ideas of latent thinking to the overleaf

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

- Can we use the dot-product of the feature vector as a measure?

- Steering projection
    - Can we identify whether this is a general machineness subspace? Like domain independent direction? And generator independet?
    - Can we zero out the domain of the steetring vector? 
        - pair innputs, why is this not already done?
        - We can ue svd on the steering vector to denoise; but then how do we reconstruct/use this vector

    - combine domains and the use svd 
    - we can use the projection score for classification; or to compare the cleaning and whether it improves separability
    - can we build the maniold with n domains, and test domains that were not used to find that space?
    - rather than projecting x on the activation vecgor, we could project into the subspace; and then get the score as the norm in the subspace

- denoise MANY steering vectors, not one?

3. "Denoising" via PCA (The "Common Signal" approach)Often, the first Principal Component (PC1) of your difference vectors is the "General Machine Style," while PC2 or PC3 are "Domain Noise."Collect many difference vectors $\mathbf{s}_i = x_{m,i} - x_{h,i}$ across many different topics.Run PCA on these difference vectors.The First Principal Component is your "Denoised Steering Vector." It represents the shared signal that exists across all domains.

- WHY: does not lok like steering is working at all?

- COE as OOD with mahambolis distance
    - formulate as a OOD task of human text similar to the neurips paper
    -  same motivation; bc OOD issues, we should especially test this in our case
    - we need a reference distributions
    - then we only need machine texts and not human texts
    - can show first with pca 

NEXT
- denoise the activations for coe
- ood gaussian fit 
    - we can have one machine and then a mix of machine attacks + normal human - > then fit gaussian on machine and rest is ood
- check bin
- check why denoising steering does not work

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

# Knowledge

- After each attention/FF block it is written to the residual stream, not after one block (see nanogpt)

- hidden states capture all proper hidden states. First is the embedding, and last is before logits.

- Magnitude change: This is the Eucledian distance between two vectors (not change in its length)
    - This is the Eucledian distance between two vectors (not change in its length)


- L2 norm = Eucledian distance = straight line difference between two vectors

- L1 norm = Manhatten distance = distance along grid lines to move between two vectors

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