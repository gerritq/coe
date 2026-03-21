# Progress

- coe: think about other features of hidden state progression
    - Added flag for norm normalization change / no norm => rm seems to lead to more separable blobs
    - prefix => for Wikipedia, it leads to differences; but not for other domains. Need to compare results of scoring model with and w/o prefix

    - difference between first and last
    - something dimension based (with squared differences)
    

- chain of logits
    - Implemented: 
        - mean/std entropy across tokens + mean/std of diff entropy across tokens
        - Full vocab vs topk
    - entropy dynamics?
    - KD div
    - ppl
    - logit vector metrics
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

- Can we do this for AFC?
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

# Story
- Idea:
    - CoE: coe discrepancies may happen when LLMs generate correct and incorrect responses
    - We hypothezise the same but for machine vs human texts, i.e., the model "thinking process" (=coe) differs between these types of text
    - Task 1: how to quantify the coe features
    - Task 2: how to use those features to predict whether a text is machine or human