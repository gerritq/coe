# To do

- run raid on the hpc, to avoid gbs

- in ood table: i think there are duplicates and they overwrite each other!!
- Check: new run for default and meta-no-pca for m4 gpt4 and yelp

1. **ID**
- TO DO:  
    - Run missing models
    - Add learn2rewrite, maybe DNA-GPT; maybe learn 2 distance
    - Keep only our models
    - Other tables comments
    - Writing 

2. **Edit distance**
- TO DO:  
    - Table comments and writing
- POTENTIAL TO DO:  
    - Reason why it not works with Beemo is that there is not variability
    - We could try their other prompts and/or Llama

3. **Layer-analysis**
- TO DO:  
    - Check that the aggregation is correct
    - Writing 

4. **Efficiency**
- TO DO
    - Check the figure file
    - Writing

5. **OOD**
- Check new defualt/meta-no-pca runs
- Figure out why BiScope is so good

6. **Descriptives**
- L1 plot is interesting as it confirms the low-dim space
- HOT!
- Some good plots: https://transformer-circuits.pub/2026/emotions/index.html
    - Maybe we can also do PCA on multiple probes?
    - Or can we do PCA on all layer probes?
        - We can plot the probes in to 2D space?s




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



99. Misc
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
- add new detector: https://github.com/ranhli/Learning2Rewrite
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

