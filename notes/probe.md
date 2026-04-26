# To do

- Evaluation threshold fixing. Not sure why the tuned threshold is so fucked.
- Implementation ideas
    1. PCA activations
    2. Code the projection idea; projection across all layers and top-k selected on the validation set
    

- Add a new "probe" folder
    - Implement the probing idea with linear regression
    - Implement the denoising idea
    - Use low-dim representations for the classifier
    - For low-dim probes, we should maybe not reguralize

- Correct the evaluation function
    - Check that this is properly implemented (see pangram optimizing f1)
    - Tune the threshold properly, not with roc


## Feature Assembly idea (Rethinking LLM-as-a-judge)

- Try the ideas of 
    - Concatenating AH and RS features
    - Mean pooling performed best in their case, we should try mean and last_token
    - PCA works best on their features, we should try how they implement it

**Misc**

- Can we construct other features from the AHs?

- Why does PC

---

## Analysis Ideas

- Performance by layer

- Performance using different surrogate models

- Performance under different data sampling strategies (comparing with training-based methods)

- OOD detection

## Brain Dump

- Can we combine PCA and projection on w as a new method?

- Try whether pooling improves performance

- Analyze at which layers probes are most performant

- Compare probes across layers

- Comparison along sample efficiency (500 samples won't suffice for Bert)