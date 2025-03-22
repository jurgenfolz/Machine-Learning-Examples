Non-Negative Matrix Factorization (NMF)
=======================================

**Non-Negative Matrix Factorization (NMF)** is an unsupervised learning algorithm used for **dimensionality reduction** and **feature extraction**. It factorizes a non-negative data matrix into two lower-dimensional non-negative matrices, preserving the additive nature of the data. NMF is widely used in text mining, image processing, and recommendation systems.

1\. Model Definition
-------------------

Given a non-negative data matrix $`X \in \mathbb{R}^{m \times n}`$, NMF decomposes it into two non-negative matrices:

$`
X \approx W H
`$

where:
- $` W \in \mathbb{R}^{m \times k} `$ is the **basis matrix**,
- $` H \in \mathbb{R}^{k \times n} `$ is the **coefficient matrix**,
- $` k `$ is the **reduced rank** (number of latent features).

The goal of NMF is to find $` W `$ and $` H `$ such that their product approximates $` X `$ while maintaining non-negativity constraints.

2\. Cost Function
----------------

NMF minimizes the reconstruction error using **Frobenius norm**:

$`
J(W, H) = \frac{1}{2} \| X - W H \|^2_F
`$

or **Kullback-Leibler (KL) divergence** for probabilistic interpretations:

$`
J(W, H) = \sum_{i,j} X_{ij} \log \frac{X_{ij}}{(WH)_{ij}} - X_{ij} + (WH)_{ij}
`$

where:
- The Frobenius norm measures the Euclidean distance between $` X `$ and $` W H `$,
- The KL divergence measures how one probability distribution diverges from another.

3\. Optimization Methods
-----------------------

NMF is solved iteratively using **multiplicative update rules** or **gradient descent**.

### Multiplicative Update Rules:

$`
W \leftarrow W \frac{X H^T}{W H H^T}
`$

$`
H \leftarrow H \frac{W^T X}{W^T W H}
`$

where element-wise division is applied.

### Alternating Least Squares (ALS):

Minimizes the objective function by fixing one matrix and solving for the other in alternating steps.

4\. Assumptions
--------------

NMF assumes:
1\. **Non-Negativity**: The input data, as well as the factorized matrices, must be non-negative.  
2\. **Additive Data Structure**: Captures parts-based representations, making it useful for applications like topic modeling and image decomposition.  
3\. **Low-Rank Approximation**: The data can be well-represented with fewer latent features.  

5\. Performance Evaluation
-------------------------

The quality of NMF decomposition is evaluated using:

- **Reconstruction Error**: Measures how well $` W H `$ approximates $` X `$.  
- **Explained Variance**: Determines how much of the data variability is captured by the decomposition.  
- **Sparsity Measure**: Encourages simpler and more interpretable representations.  

6\. Extensions
-------------

- **Sparse NMF**: Adds L1 regularization to encourage sparsity in $` W `$ and $` H `$.
- **Graph-Regularized NMF**: Incorporates structural information from graphs.
- **Non-Smooth NMF (nsNMF)**: Introduces a smoothing term to reduce noise sensitivity.

7\. Pros and Cons
----------------

### Pros:
1\. **Interpretable**: Produces non-negative, parts-based representations.  
2\. **Dimensionality Reduction**: Extracts meaningful low-dimensional structures.  
3\. **Handles Large Datasets**: Efficient for large-scale data.  
4\. **Useful for Clustering**: Can reveal hidden topics or components in data.  

### Cons:
1\. **Local Minima Issues**: Convergence depends on initialization.  
2\. **Sensitive to Noise**: Noisy data can lead to poor decomposition.  
3\. **Requires Tuning**: The choice of $` k `$ and regularization parameters affects results.  
4\. **Not Always Unique**: Different factorizations can yield similar approximations.  
