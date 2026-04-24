# To do

- Add a new "probe" folder
    - Implement the probing idea with linear regression
    - Implement the denoising idea
    - Use low-dim representations for the classifier
    - For low-dim probes, we should maybe not reguralize

- Correct the evaluation function
    - Check that this is properly implemented (see pangram optimizing f1)
    - Tune the threshold properly, not with roc

## Misc

- Try whether pooling improves performance

- Analyze at which layers probes are most performant

- Compare probes across layers

- Comparison along sample efficiency (500 samples won't suffice for Bert)