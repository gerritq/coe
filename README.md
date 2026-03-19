# To do

- chain of logits
    - KD div
    - ppl
    - cross entropy change
    - top-k tokens distribution
    - logit vector metrics
    - cross-entropy
    - total variation distance 
    - JS divergence



- for the ration, do we need to normalize?
- spikes at the beginning and end, should we exclude them?
- confirm pooling is correct
- difference between layers
- rename layer profile trajectory; add layer nums
- identify other metrics
- try the karpathy agent on my task

# Brain dump

- Prefix: can we add is this text machine or human generated?
- Can we use the uncertainty at the unembedding layer across tokens?
- Dimension indpendent metric as in OOD paper
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


# Qs
- Why is there no difference in reddit?
- Why magnitude change if it is distance change?

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
