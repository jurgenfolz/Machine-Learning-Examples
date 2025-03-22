Manifold Learning with t-SNE
============================

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a non-linear dimensionality reduction technique used for **visualizing high-dimensional data** in a low-dimensional space (typically 2D or 3D). It is particularly well-suited for discovering **manifold structures** and revealing clusters or patterns in complex datasets.

1\. Model Definition
-------------------

Given a high-dimensional dataset $`X = \{x_1, x_2, ..., x_n\}`$, t-SNE constructs a **probability distribution** over pairs of high-dimensional objects such that similar objects have a high probability of being picked, and dissimilar ones have a low probability.

In the low-dimensional embedding $`Y = \{y_1, y_2, ..., y_n\}`$, it then tries to match the same distribution using a **Student-t distribution** with one degree of freedom.

### Step-by-step process:

1. Compute pairwise similarities in high-dimensional space using Gaussian kernels:

$`
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
`$

2. Compute the symmetric joint probabilities:

$`
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
`$

3. Define a similar probability distribution $`q_{ij}`$ in the low-dimensional space using a Student-t distribution:

$`
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
`$

4. Minimize the **Kullback-Leibler (KL) divergence** between the two distributions:

$`
C = KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
`$

2\. Cost Function
----------------

The cost function used in t-SNE is the **KL divergence** between the high-dimensional pairwise similarity distribution $`P`$ and the low-dimensional similarity distribution $`Q`$:

$`
C = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
`$

This objective ensures that similar data points in the high-dimensional space stay close together in the low-dimensional embedding.

3\. Optimization
----------------

t-SNE minimizes the KL divergence using **gradient descent**. To ensure stability and quality of the embedding, t-SNE typically applies the following techniques:

- **Early Exaggeration**: Temporarily increases $`p_{ij}`$ values to form better-separated clusters in early iterations.
- **Momentum**: Speeds up convergence and avoids poor local minima.
- **Perplexity**: A key parameter that balances attention between local and global data structures.

4\. Assumptions
--------------

t-SNE assumes:
1\. **Local Structure Matters**: Emphasizes preserving local neighborhoods over global geometry.  
2\. **Data Lies on a Manifold**: Suitable for complex, non-linear relationships.  
3\. **Distance Reflects Similarity**: Pairwise distances are assumed to encode meaningful similarity.  

5\. Performance Evaluation
-------------------------

t-SNE is typically evaluated visually, rather than through formal metrics, by assessing:

- **Cluster Separation**: Well-separated clusters indicate meaningful embeddings.
- **Preservation of Local Structure**: Neighborhoods in high-dimensional space are preserved in low-dimensional space.
- **Reproducibility**: Stable results across different runs (with consistent initialization and parameters).

6\. Extensions
-------------

- **Parametric t-SNE**: Uses a neural network to learn a parametric mapping.
- **Barnes-Hut t-SNE**: An approximation algorithm that reduces computational complexity to $`O(n \log n)`$.
- **FIt-SNE**: Fast interpolation-based variant suitable for very large datasets.

7\. Pros and Cons
----------------

### Pros:
1\. **Reveals Complex Structure**: Excellent for visualizing clusters and non-linear relationships.  
2\. **Non-Linear Embedding**: Captures manifold geometry effectively.  
3\. **Widely Used in Practice**: Popular in bioinformatics, NLP, and computer vision.  
4\. **Intuitive Visualization**: Makes high-dimensional data interpretable.  

### Cons:
1\. **Computationally Expensive**: Slower on large datasets.  
2\. **Not Scalable to New Data**: Embedding must be recomputed for new samples.  
3\. **Parameter Sensitivity**: Perplexity and learning rate must be carefully tuned.  
4\. **No Global Distance Meaning**: The scale of distances in the embedding is not globally meaningful.  

t-SNE is a powerful tool for exploring and visualizing high-dimensional data, especially when understanding the structure or clustering behavior of data is the primary goal.
