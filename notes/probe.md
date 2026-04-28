# To do

**Primary**
- add Repguard metrics
- Create a a ID table
- Add BiScope
- Add DNA-GPT
- Add DetectGPT

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


## Feature Assembly idea (Rethinking LLM-as-a-judge)

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

---

To show whether trained probing vectors are similar

- Run the Cosine of vectors across layers between two domains/languages (vis as a heat map)

---


# Main Analysis

## ID 

- Compare against various baselines, but not necessarily trained on the data (zero-shot, ML)

## OOD

- Compare against trained model on the data, for those we can then show AUROC/TPR

---

### Additional Analysis

- Can use different surrogate models

- Compare trained concept vectors across languages and domains
    - Can be done for each layer

- Performance by layer

- Performance using different surrogate models

- Performance under different data sampling strategies (comparing with training-based methods)

- OOD detection

- Can we use the projection score to measure the degree of editing as in editlens?

- Efficency analysis against bert models (compute + data)

## Ablations

- pca vs no pca

- number of components (I think there is a way to determine this)

- number of layers to consider? (top-k layers)

## Brain Dump

### Aggregations

- Top-k best performing on the val set? Or top-k most correct and certain?

- Read papers on aggregation methods; we do not need the projection idea

- Can we do a moving average/momentum like the EME aggregation, just with the performance metric as the weight?

### Misc

- Can we also run the logistic regression and then see how many neurons survive? This would further strengthen the argument that we need to operate in low-dim spaces.

- Can we combine PCA and projection on w as a new method?

- Try whether pooling improves performance

- Analyze at which layers probes are most performant

- Compare probes across layers

- Comparison along sample efficiency (500 samples won't suffice for Bert)

- For the low-dim representation, should we not penalize?


# Paper/Posts

## https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes

