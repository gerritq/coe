# Representation Quality Analysis

Skean et al. (2025)
- Investigate layer by layer performance of models across tasks
- They do not study ID
- To analyze layers, they combine three perspectives

    1. Information-theoretic: How much do layer compress or preserve information?

    2. Geometric: How do token embeddings unfold in high-dimensional space? 
        - How are representations structured in high-dimensional space?
        - How are representations organized geometricall?


        - Classic approaches include analyzing the rank and singular values of the representation matrix
        - Anisotropy has been used to study compression
        - Curvature quantifies how smoothly tokens are mapped across consecutive positions

    3. Invariance: Are embeddings robust to input perturbations?
        - Evaluates how well representations support downstream goals.
        - Augmentation based approaches estimate invariance to perturbations
        - In CV, these methods correlate with downstream performance

- They show that these metrics appear distinct, but can be unified to explain the balance of compression, geometry and invariance

    - They can be unified with matrix-based entropy

        - 



## Information Content

### Entropy

**Intuition**
- Low:
    - Representations are compressed
- High:
    - Representation is spread out across many principal directions

## eRank

## Geometric Structure
## Anisotropy

**Intuition**
    - Anisotropy has been used to study compression

## ID
- Tulchinskii et al. (2023) show that MGT has a lower ID. Form their paper 
    - TO DO

- ID describes the minimum number of features to represent the data (Skean et al, 2025)

---

# Notes

- Mention that information-theoretic and geometric metrics offer complementary perspective on representation quality
- Beyond confirmining that MGT cn be detected in latent space, we try to elucidate __why__ --- this is strong and should be placed 1-3 times
- We should add what each metric is answering!
- For each perspective, we should have one takeaway
- Mention the last token embedding of llama across layers, which is R l Nxdl which we analyse for Wikipedia in M4GT --- in appendix we should that these findings generalize across domains and other benchmarks.  