Principal Component Analysis (PCA)
==================================

**Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional representation while preserving as much variance as possible. PCA is widely used in exploratory data analysis, visualization, and preprocessing for machine learning models.

1\. Model Definition
-------------------

Given a dataset $`D = \{x_i\} `$ where $`x_i `$ is a feature vector in $`\mathbb{R}^d `$ (a $`d`$-dimensional space), PCA aims to find a set of orthogonal **principal components** that maximize the variance in the data.

Each principal component is a linear combination of the original features:

$`
z_j = w_j^T x
`$

where:
- $` w_j `$ is the weight vector (eigenvector) corresponding to the $`j`$-th principal component,
- $` z_j `$ is the transformed data along the $`j`$-th principal component.

The goal is to find a new basis in which the first principal component captures the most variance, the second captures the second-most variance, and so on.

2\. Cost Function
----------------

PCA minimizes the **reconstruction error**, defined as:

$`
J(W) = \sum_{i=1}^{n} \| x_i - \hat{x}_i \|^2
`$

where $` \hat{x}_i `$ is the reconstructed version of $` x_i `$ using a reduced number of principal components.

Alternatively, PCA can be formulated as maximizing the **variance**:

$`
\max_W \frac{1}{n} \sum_{i=1}^{n} (W^T x_i)^2
`$

subject to the constraint that $` W `$ is an orthonormal matrix.

3\. Eigenvalue Decomposition
----------------------------

PCA finds the **eigenvectors** and **eigenvalues** of the covariance matrix:

$`
\Sigma = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^T
`$

where $` \bar{x} `$ is the mean of the dataset.

The eigenvectors (principal components) define the new feature space, and the eigenvalues indicate the amount of variance each principal component explains.

### Steps for PCA:

1. Compute the **covariance matrix** $` \Sigma `$.
2. Perform **eigenvalue decomposition** of $` \Sigma `$.
3. Select the **top $k$ eigenvectors** corresponding to the largest eigenvalues
