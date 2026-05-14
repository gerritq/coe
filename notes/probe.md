# To do

- change GEScore to GECScore and Revise is zero-shto
- name xsum -> news
- change tsm names

- change name yelp uppercase

- change tsm names
- add revise to zero-shot
- AUC in plots not AUROC
- add repre to supervised


- Read: https://www.goodfire.ai/research/manifold-steering#
    - And their other paper

- Check new meta pca yelp
- ef rank missings
- meta sizes attention issue


- Write rel work + experimental setup + ID results with GPT
- Read and get intuition for metrics 
    - e-rank: 

RUNNING
    - ID
        - 33886967 Raidar
    - Sample
        - 33889365 meta no pca
    - Training size
        - TO DO: Repre fix
        - BiScope 33879988
        - Beautify: renaming; no bench; Raid

    - Qual metrics:
        - Effective Rank seed 46: 33889209 
        - Create a plot with CI


---

1. **ID**
- TO DO:  
    - Run missing models
    - Other tables comments
    - Writing 

---

2. **Edit distance**
- TO DO:  
    - Table comments and writing
- POTENTIAL TO DO:  
    - Reason why it not works with Beemo is that there is not variability
    - We could try their other prompts and/or Llama
    - Could add a baseline

---

3. **Layer-analysis**
- TO DO:  
    - Update to new datasets and check that the aggregation is correct
    - Writing 

---

4. **Efficiency**
- TO DO
    - Hanqi: add confidence intervals! (run with 5 different seeds)
    - Check the figure file
    - Writing

---

5. **OOD**
- TO DO
    - Add Biscope 
    - Writing
    - Check new defualt/meta-no-pca runs

---

6. **Descriptives/The why**

    - HOT: Can we do something similar as in https://arxiv.org/pdf/2502.14888 see section 3.1

    - read rel work: https://arxiv.org/pdf/2502.02013

    - check: LANGUAGE MODELS REPRESENT SPACE AND TIME

    - HOT: Can we use feature activations as in: When Truthful Representations Flip Under Deceptive Instructions?

        - Llama scope: Llama Scope

    - We can make the same linearity claim as in: https://arxiv.org/pdf/2509.10625?

    - Universal claim: https://aclanthology.org/2025.findings-acl.38.pdf

    - Representation quality metrics: LATENT THINKING OPTIMIZATION: YOUR LATENT REASONING LANGUAGE MODEL SECRETLY ENCODES REWARD SIGNALS IN ITS LATENT THOUGHTS

    - Heatmap of probing vectors: https://transformer-circuits.pub/2026/emotions/index.html

    - L1 plot is interesting as it confirms the low-dim space

    - HOT!
        - Some good plots: https://transformer-circuits.pub/2026/emotions/index.html
            - Maybe we can also do PCA on multiple probes?
            - Or can we do PCA on all layer probes?
                - We can plot the probes in to 2D space?s

    - Hanqi idea: increase linearity
        - Check the results for the polynomial
        - Check the polynomial paper and how they dealt with high dims
        - Run mlp with different depth

    - Hanqi: 
        - Run the rank test on clean m4 data

    - Idea of rank of weight matrix:
        -https://aclanthology.org/2020.emnlp-main.254.pdf see section 2.3
        - https://aclanthology.org/N19-1419.pdf
        - or how many weights are used for human vs machine text

    - Polynomials do not seem to work: check the polynomial paper to see what they do in high-dimensions

    - There is this within variance idea, somewhere I have read it.


2. **Ablations**
- We could keep PCA and then they we try but it does not work; we were motivated to use PCA componensts base don the findings that activations are separable in low-dimensions; but we did not find any imoprovement. We believe this is because the logistic regression with l2 penalty performs feature selection anywyas.
- Check PCA one last time; attn does not make sense, this is a different space!
- Confirm that PCA does not benefit for OOD

- Further ablations
    - Layer
        - default/pca: linear probe vs non linear MLP (strong evidence for linearity if the same)
        - default vs pca across values
        - default/pca: mean vs last token pooling
        - default/pca: mean aggregation vs weighted aggregation (try F1)
    - Meta
        - pca: Attention vs no attention
        - No attention: default vs pca values

67. Hanqi Feedback
    
    1.  Can we understand why LP works so well OOD?
        - Can we see increase non-linearity and seeh whether it is getting worse?
        - x-axis is non-linearly of probes - and y-axis performance => so non linear
        - MLP with or various activations or probe with higher exponents
        - Test this for OOD

    2. Ablations with first and last layer

    3. For sample efficiency: add CI, run with different seeds

    4. Combine datasets, apply pca
        - Get SV and the plot
        - Or show the decision boundary bc orthogonal

70. Writing
- CHECK TABLE AGGREGATIONS AND DUPLICATES
 - For probing generalization and performance: https://arxiv.org/pdf/2511.17408

--- 

99. Misc

    - For cross benchmark https://arxiv.org/pdf/2509.10625

    - ensure in all tables that we take the mean not weighted

    - Probing vector similarity plot
        - Could be a nice argument for universal, rather than relying on the numbers; to show this also visually

    - rerun encoder, biscope, and repreguard on the four sample efficency data sets

    - M4 generators: are we mixing the data? Should check if we wanna use it

    - Maybe another dataset to consider: https://aclanthology.org/2024.findings-naacl.29.pdf

    - ensure we use mean projection metrics
        - compare to weighted metrics

    - We ran Repre with val, rerun with training split

    - The L1 penalty regression is still an interesting case which I would like to see somewhere

    - ensure that tables are pulling the correct numbers

    - fix repre scores

    - fix PCA None

    - can extend beemo to llama, but for now we do gpt4

    - norm scores?

    - We could to the CB if we have space

    - Add cool symbols top the table for reddit etc.


## Has to be done


- Generator ds using M4!!! In appendeix, we use it for CB

- Once and for all, correct the evaluation metrics. 
    - Acc/F1 do tuning and Yuoden's k
    - ALso TPR@1FPR has to be done on the val set

- Clean probe code to have a single flag which runs the specific setting

- Checks
    - Repreguard metrics
    - Try to run fluoscope with the old model
    - MLModel perform crap

- entropy norm?
- read: Context Matters: Analyzing the Generalizability of Linear Probing and Steering Across Diverse Scenarios


---
---
---

# LP Tweaks

- Normalize projection scores with min-max (I believe EdtiLens does the same)

- Test other metrics for the performance aggregation. Higher strength or AUROC does not seem to improve much.

- For the meta classifier, try also using input from attention heads, as in "Rethinking LLM-as-a-judge"

    - https://aclanthology.org/2026.eacl-long.324.pdf also investigate heads. Then we could also identify which heads are important

- Try different performance weight taus; try different pca components; can we try other metrics?

- Attention probes: https://learnmechinterp.com/topics/attention-probes/

---

# Feature Assembly idea (Rethinking LLM-as-a-judge)

- Try the ideas of 
    - Concatenating AH and RS features
    - Mean pooling performed best in their case, we should try mean and last_token
    - PCA works best on their features, we should try how they implement it

**Misc**

- Can we construct other features from the AHs?

---

# Analysis Ideas

## Descriptives

**A. To show the low-dimensionality**

- Similar plot as in "Mitigating overthinking" where we show the separability across layers

- PCA by layer, then share of components by number of components (the Manifold steering plot)

    - Here, search how to identify the optimal number of components and pick it.

- Run L1 regularization probes, and see how many neurons are zeroed out
    - Could be done per layer

- Can we do one features (domain, generator) etc visualizations in 3D? As in Pangram

    - We could use the M4 dataset and use their domains, languages, and generators to visualize in 3D

    - Can use t-sne and clustering



**B To show whether trained probing vectors are similar**

- Run the Cosine of vectors across layers between two domains/languages (vis as a heat map)

- Other ways to measure intrinsic dimensions?

**C Benchmark difficulty**

- Can we use the trained probing vectors, and rank benchmarks with them for difficulty? That is the more their machine/human data is pointing in the corresponding direction, the easier to separate?

    - This could be a nice plot

## Main Analysis

### ID 

- Compare against various baselines, but not necessarily trained on the data (zero-shot, ML)

### OOD

_Note _: Geometry of truth paper is one of the first to show that probin directions are universal.

- Compare against trained model on the data, for those we can then show AUROC/TPR

---

## Additional Analysis

- Visuslization as in Linear Representation of Political. ..

- Can we steer the text into making it more human like, and then fool detectors?

- Can use different surrogate models

- Compare trained concept vectors across languages and domains
    - Can be done for each layer

- Performance by layer

- Performance under different data sampling strategies (comparing with training-based methods)

    - We can do Roberta vs ours
    - We can create a plot where we compare the OOD of each dataset, LP vs Roberta

- OOD detection

- Can we use the projection score to measure the degree of editing as in editlens?
    - May also use Beemo data

- Efficiency analysis against bert models (compute + data)

- Same as below: can probes detect the degree of AI rewriting? Use editlens

- Can we do an analysis on the similarity of probing vectors?

## Ablations

- pca vs no pca

- number of components (I think there is a way to determine this)

- number of layers to consider? (top-k layers)

- Aggregation mean vs last last token

- Efficiency
    - Latency, flops, and parameters as in LLM Safety from within

---

## Brain Dump

### Aggregations

- Top-k best performing on the val set? Or top-k most correct and certain?

- Read papers on aggregation methods; we do not need the projection idea

- Can we do a moving average/momentum like the EME aggregation, just with the performance metric as the weight?

- We do a similar performance aggregation as in the SIREN paper

### Misc

- Good ideas here including sparsity of probes: Activation Steering With Mean Response Probes : A Case Study In Suppressing Sycophancy In Language Models During TTC

- Can we use attention probes? https://learnmechinterp.com/topics/attention-probes/

- Can we also run the logistic regression and then see how many neurons survive? This would further strengthen the argument that we need to operate in low-dim spaces.

- Can we combine PCA and projection on w as a new method?

- Try whether pooling improves performance

- Analyze at which layers probes are most performant

- Compare probes across layers

- Comparison along sample efficiency (500 samples won't suffice for Bert)

- For the low-dim representation, should we not penalize?

- We can also try linear probe ensemble from multiple small models as in the Siren paper

- Can we do a cross dataset comparison? **That could be novel**


# Paper/Posts

## Mech inter

LHR: https://learnmechinterp.com/topics/linear-representation-hypothesis/

Probing: https://learnmechinterp.com/topics/truthfulness-probing/#truth-directions-transfer-across-topics

Probing: https://www.anthropic.com/research/probes-catch-sleeper-agents

## Lower dimensions

[HF](https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes)

- They call it "localized subspace"

[Famous mediation paper](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction?utm_source=chatgpt.com)

To better understand the representations of MGT and HWT, we perform PCA decomposition

# Qualitative metrics

## Isotropy

- Definition: the property of being uniform in all directions

- LLMs: geometric state where the data points/embeddings are uniformly distributed in all directions within the space

- When isotropic, the variance of an embedding is spread across all dimensions

- Cosine similarity of unrelated tokens is close to zero


## Anisotropy

- Most LLMs "suffer" from anisotropy

- Instead of utilizing the full vector space, embeddings occupy a narrow, cone-shaped region

    - Why cone-shaped: embeddings scatter along one-dominant region (in 2D, instead of a circle as in isotropic spaces, the anisotropic space is an ellipse and a cone in higher d)

    - This means a few-dominant directions explain much of the variance

- Cosine similarity of unrelated tokens is larger than zero

## Effective Rank

Idea: 
    - rank-based metric, they introduce Diff-rank = differences between effective ranks
    - Metric focused on the representations, different from the loss. Hidden states capture semantic and syntactic information of sentences
    - How efficiently the model eliminates redundant information during training

Intuition: 
    - Information-theoretic measure: provides a measure for quantifying noise reduction in LLMs
    - Rank provides two measures: 
        - Geometric: linear independence/effective dimensions in the representation space
        - Information: amount of information contained in the representations
            - __Key__: a lower rank indicates that information has been structured or compressed

    - From Latent thinking:
        - How effectively the model extracts key concepts and reduces noisy features in its latent representations.
        - A higher effective rank implies noisier features
        - A lower effective rank implies better noise reduction and more compact representations
