# Story

- Should it maybe be
    - Decision boundary is linear --- but why


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

- They group entropy to compression and curvature,rank,id to geometry; but the "Layer Optimization" paper does it differently



## Information Content

### Entropy

High entropy / high effective rank means the variance is distributed across many singular directions; low values mean a few directions dominate

**Intuition**
- Low:
    - Representations are compressed
    - Prune redundancy
- High:
    - Representation is spread out across many principal directions
    - Preserving essential distinctions

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

## Analysis

### Information-theoretic measures


Takeaway: 
    - HWT representations are more complex and utilize more of the available representation space than machine text
    - HWT representations are more information-rich and higher-rank
    - MGT collapse into lower-rank and are more compressed


Observation: Both entropy and effective rank are substantially higher for human than for machine text, across layers

Intuition: Both metrics measure how "spread out" the spectrum of the representation covariance is. 

Interpretation:
    - MGT
        - Representations are more spectrally concentrated -> informationm is compressed into fewer dominant directions
        - Encodes less variance
        - Representations consists of more regular, repetitive, or lower-diversity activation patterns
        - Representations use a small subspace to present information (likely because the underlying linguistic patterns are more predictable, uniform and structurally constrained compared to rich, less predictable human writing)
        - MGT representation information is captured by a small number of components

    - HWT
        - Representations use a broader and more distributed representational subspace
        - Representations carry more information
        - The information/(variance) is spread across more independent directions -> indicates a richer, less compressed, and less redundant structure
        - Representations exhibit a richer mix of features

- Summary: 

### Geometric measures

Takeaway: 
    - HWT occupies a geometrically richer and higher-dimensional manifold
    - MGT is more directionally concentrated and lower-dimensional

Intuition:
    - Anisotropy measures how directionally biased the cloud of representations is — high anisotropy means the vectors cluster along a preferred axis (a narrow cone).
    - Intrinsic dimensionality measures the dimensionality of the manifold the points actually live on. 

Observation:
    - Machine representations are more anisotropic, especially from layers 8-25
    - Human representations have substantially higher ID across layers

    - MGT
        - Anisotropy (high)
            - Activations are dominated by a few directions
            - Representations cluster along a narrow cone in latent space
            - Geometrically, representations are more aligned, more clustered, and less evenly distributed in latent space

        - ID (low)
            - Representations live on a simpler lower-dimensional manifold

    - HWT
        - Anisotropy (low)
            - Representations occupy a more isotropic space
            - Representations are spread out more evevenly in space

        - ID (Low)
            - Representations vary along more independent degress of freedom (consistent with HWT being more diverse in style, structure etc.)
            - Representations live on a higher-dimensional, complex manifold

- Summary: Geometrically, machine text is squashed into a dense, directional cluster on a simpler manifold, whereas human text remains scattered more evenly across a complex, high-dimensional space.

### Linear separability

- Metrics explain global differences of machine and human representations --- but they do not directly show that they are linearly separable

- Help to explain why the linear separability emerge

- These metrics can provide a plausible mechanistic explanation for why linear boundaries work

Hi Hanqi, about your comment "How about this logic: firstly, we just blindly (not real blind) evaluate existing latent metrics (so we need to justify they are from different aspects), then the evaluation results point out, they could be linear separated."

How about we reverse the logic to 
1. We show machine and human representations are linearly separable (phenomenon), 
2. We ask what makes representations different? (research question)
3. We use 4 metrics to analyze human/machine representations (analysis)
4. We conclude that they occupy different subspaces, which simple hyperplanes exploit (answer)

__Logic__

1. MGT and HTW are linearly separable in latent space
    - Prior work shows high detection from linear classifiers: RepreGuard and Tulchinskii et al. (2023)
    - Our low-dimensional projections show the separation holds across domains
    - Our ablation: increasing MLP non-linearity __hurts__ probe performance — the true decision boundary of human/machine text is linear, and adding complexity overfits to noise


2. However, this leaves the key question unanswered: __RQ__: How do machine and human representations differ in latent space? What properties of the representations make them linearly separable?

3. Representation quality analysis reveals two complementary patterns:
   
   - **Information-theoretic** (entropy, effective rank): human/machine distribution shave different eigenspectra
        - Human text representations spreads information across many directions 
        - Machine text representations information are compressed 
   
   - **Geometric** (anisotropy, intrinsic dimensionality): human/machine distributions have different shapes (manifold) and locationsß 
        - Human text representations lie on a more complex, higher-dimensional manifold & on which it is more uniformly distributed (isotropic)
        - Machine text representation lie on a simpler, lower-dimensional manifold & on which they are more concentrated along few dimensions (anisotropic)

4. Answer to RQ

- These findings offer a plausible explanation of why linear separability works: 
    
    - Human-written text occupies a more complex and isotropic high-dimensional "cloud" spanning the model's latent space. Conversely, machine-generated text collapses into a more anisotropic, low-dimensional, tightly packed "cone" or subspace.

    - Therefore, a simple linear hyperplane is geometrically sufficient to isolate the machine subspace
    
    - However, while these metrics are consistent with the identified linear separability and provide a plausible mechanism, we do not claim that they prove it.

---

# Notes

- "These results do not establish linear separability directly, but they help explain why simple linear probes can distinguish human- and machine-generated text."
- Mention that information-theoretic and geometric metrics offer complementary perspective on representation quality
- Beyond confirmining that MGT cn be detected in latent space, we try to elucidate __why__ --- this is strong and should be placed 1-3 times
- We should add what each metric is answering!
- For each perspective, we should have one takeaway
- Mention the last token embedding of llama across layers, which is R l Nxdl which we analyse for Wikipedia in M4GT --- in appendix we should that these findings generalize across domains and other benchmarks.  

- Wording as in Plank paper: Our findings in §5 reveal a surprisingly
coherent internal mechanism: refusal directions are not specific to individual languages but generalize
effectively across both high-resource and low-resource languages.