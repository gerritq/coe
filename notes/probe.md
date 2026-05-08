# To do
- probe_meta_no_pca_domain_attack all id and ood
- run missing OOD:
    - 33781446 run domain for all non-default => check whether there is __clear__ OOD improv
- check beemo 
- check others may have forgotten

1. **Edit distance**
- Next: Run Beemo data set creation on the HPC
    - Add a histogram of edit strength using the metrics (either main or appendix)
- Table of edit strength
- Check M4 results but it does not seem to work

2. **Ablations**
- PCA does not seem to help, neither does attention
- Check whether this is true for OOD
    - Currentl running OOD with meta_no_pca

- Create a table for the pca results
- Further ablations
    - Layer
        - default/pca: linear probe vs non linear MLP (strong evidence for linearity if the same)
        - default vs pca across values
        - default/pca: mean vs last token pooling
        - default/pca: mean aggregation vs weighted aggregation (try F1)
    - Meta
        - pca: Attention vs no attention
        - No attention: default vs pca values

3. **Efficiency**
- Use ablation scrupt to run meta and layer for different sample sizes
- Consider which sets and whether to include a comparison

4. **OOD**
- Consider which baselines to show + whether to show meta probe g
- Beautify plot



99. Misc
- fix repre scores
- fix PCA None
- can extend beemo to llama, but for now we do gpt4




- entropy scaled run results
- fix roberta
- ai degree editjung data


1. Run CB with M4 for probing and roberta   
- Run APT with M4

2. Update ID table and OOD figure
- rename detectrl
- run the ATP with meta; add results

3. Other runs
- Run APT with M4

## Today 

1.  Code the OOD setupt where one vs all
    - Select Roberta + 3 other trained methods for this vs ours
    - Need to check bert results, too good overfitting?

2. Run our first LP baseline
    - DONE Add the attention features to the meta probe
    - Add summaries like mean etc.
    - Begin ablations

5. Universality of probing vectors
    - Take a dataset with varying domains, languages, and generators.
    - Run full OOD 
    - Obtain the probing vectors across those settings and show their similarity in a heatmap

6. Descriptives
- Check the descriptives code and run
- Add human subset per domain in the t-sne plot
- Add the pca per layer line plot

## Has to be done

- Do we need to scale entropy? Because with longer sequences we will have higher entropy
    - I think yes

- Add cool symbols top the table for reddit etc.

- Supervised detectors
    - Editlens data and hf model
    - DNA GPT
    - Shared task detector

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

