# To do


I want to write a paper on using linear probes for LLM-generated text detection. I want to submit it as a long paper to EMNLP.

So far, my story and results are:

1. Introduction: 
- LLM have become very good in text generation, however, their misuse introduces concerns regarding fake news, malicious content generation, or plagiarism. Due to the mass and difficulty of identifying LLM generated text, reliable machine generated text detection (MGT) has become an active area of research.

- Existing detectors can be grouped into zero-shot and supervised-fine-tuned. Within each training paradigm, most detectors logit-based or rewrite based.

- Emerging theoretical and empirical evidence of the linear representation hypothesis which stipulates that high-level, interpretable conceptss are represented linearlt in the represenation space of LLMs. The idea is that probing vectors can be derived by contrastive datasets to identify these linear direction in the representation space for detection or intervention of the concept of interenst. This hyoptehsis has been show to hold for various fields such as language (Park et al, 2024), truth (Li et al, 2023; Marks and Tegmark 2024) or refusal (Arditi et al. 2024).

- Recent work in MGT detection has exploited this work (). Tulchinskii et al. (2023) show that MGT and human-wirtten text show can be seprated accorfing ot their intrinsic dimnesionality. Yu et al. (2024) identify intrinsic features of MGT by trhough idenitfying the distributionally most distint hiddne layer for featuer classification.
Recently, Repreguard (Chen et al, 2025) show that LLMs hiddne states activation differt foir MGT and human-generated text (HGT), and that unsuervised probes can be an effective detection method.

- We extend this line of work.

2. Methodology

- Explain how we use linear probes for MGT detection.


3. Experimental Setup

    1. In-domain detection results on DetectRL, Multisocial, TSM-Bench, and Beemo



- Aggregation single probe: pca, the weight and concat (if 0, drop)
    - Run all combos for ablation comparison
- Consider Beemo or another rewriting/paraphrasing data
- Check editlens correlation
- Run with OOD of combined data

---
- Emsemble of models from smaller models (Siren paper)
- Skean paper in LLM safety
- For sample efficiency
    - Compare Roberta vs ours
    - Compare for OOD of each dataset! Ie test vs all
- cross dataset
- OOD can also be a combination, maybe making the tables looks easier
- Compare ID and OOD perforamnce by layer
    - OOD seems to reveal different patterns
- Generate the proibing vector simiarities
    - Can we do seperate PC's tho?
- find more emnlp/acl work that uses probes
- latency, flops, and parameters as in LLM Safwty from within
- read the llm safety, we do a similar aggregation
- OOD on languages, generators, domains, and tasks?
- it may make sense to first do the probe compairson, and from there hypothesie a "universal" direction which leads to higher OOD; then confirm this in experiments

- method description as in political probes
- text fluo: single layer probes are brittle
- repreguard: unsupervised methods

**Primary**
- Add the target data to the args
- fix roc auroc metric in repreguard
- fix a ID table sith bold and second

- Add BiScope -> test (feels like we need to make it faster with batch inference)
- Add DNA-GPT
- Add DivScore


Probes baselines (report all, select later)
- add attention features
- Weighting by auroc
    - Check whether we can try other metrics
    - try different values of tau + different values of components (implement as cli)
    - Check whether it performs better than equal weighting
- try weighting the scores by probing metric
- train a meta probe
- normalize score of the projections with min max?
- Try to fit pca on collapsed layer, that can be used for easier comparison of domanins
    - But it needs to perform as well as the non-collapsed version


**Secondary**
- fix MLModel results
- fix fluo with old model

- Evaluation threshold fixing. Not sure why the tuned threshold is so fucked.
- Implementation ideas
    1. PCA activations
    2. Code the projection idea; projection across all layers and top-k selected on the validation set
    
- Correct the evaluation function
    - Check that this is properly implemented (see pangram optimizing f1)
    - Tune the threshold properly, not with roc
    - edit lens also uses min max scaling

-read Truth Directions Transfer Across Topics in https://learnmechinterp.com/topics/truthfulness-probing/
    - universality should be used by us as well

---

# Feature Assembly idea (Rethinking LLM-as-a-judge)

- Try the ideas of 
    - Concatenating AH and RS features
    - Mean pooling performed best in their case, we should try mean and last_token
    - PCA works best on their features, we should try how they implement it

**Misc**

- Can we construct other features from the AHs?

- Why does PC

---

# Analysis Ideas

## Descriptives

To show the low-dimensionality

- PCA by layer, then share of components by number of components (the Manifold steering plot)

- Run L1 regularization probes, and see how many neurons are zeroed out
    - Could be done per layer

To show whether trained probing vectors are similar

- Run the Cosine of vectors across layers between two domains/languages (vis as a heat map)

- Other ways to measure intrinsic dimensions?

## Main Analysis

### ID 

- Compare against various baselines, but not necessarily trained on the data (zero-shot, ML)

### OOD

- Compare against trained model on the data, for those we can then show AUROC/TPR

---

## Additional Analysis

- Visulaization as in Linear Representation of Political. ..

- Can we steer the text into making it more human like, and then fool detectors?
- Can use different surrogate models

- Compare trained concept vectors across languages and domains
    - Can be done for each layer

- Performance by layer

- Performance using different surrogate models

- Performance under different data sampling strategies (comparing with training-based methods)

- OOD detection

- Can we use the projection score to measure the degree of editing as in editlens?
    - May also use Beemo data

- Efficiency analysis against bert models (compute + data)

- Same as below: can probes detect the degree of AI rewriting? Use editlens

## Ablations

- pca vs no pca

- number of components (I think there is a way to determine this)

- number of layers to consider? (top-k layers)

- Aggregation mean vs last last token

---

## Brain Dump

### Aggregations

- Top-k best performing on the val set? Or top-k most correct and certain?

- Read papers on aggregation methods; we do not need the projection idea

- Can we do a moving average/momentum like the EME aggregation, just with the performance metric as the weight?

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


# Paper/Posts

## https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes

