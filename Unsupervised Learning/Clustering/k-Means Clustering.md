k-Means Clustering
==================

**k-Means Clustering** is an unsupervised learning algorithm used to partition a dataset into **k distinct clusters** based on feature similarity. It aims to minimize the distance between data points and their assigned cluster centroids, making it a widely used method for exploratory data analysis, customer segmentation, and image compression.

1\. Model Definition
-------------------

Given a dataset $`X = \{x_1, x_2, ..., x_n\}`$, where each $`x_i \in \mathbb{R}^d`$, the goal of k-means is to partition the $`n`$ data points into $`k`$ clusters $`C = \{C_1, C_2, ..., C_k\}`$ such that:

- Each data point belongs to the cluster with the nearest **centroid**.
- The objective is to **minimize intra-cluster variance** (or equivalently, the sum of squared distances from each point to its cluster center).

The centroid $`\mu_j`$ of cluster $`C_j`$ is computed as:

$`
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
`$

2\. Cost Function
----------------

The objective function of k-means is to minimize the **within-cluster sum of squares (WCSS)**:

$`
J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
`$

This function measures the total variance within all clusters. A lower value indicates tighter and more compact clusters.

3\. Optimization Procedure
--------------------------

k-Means uses an **iterative refinement algorithm**, consisting of the following steps:

1. **Initialization**: Choose $`k`$ initial centroids (randomly or using methods like k-means++).  
2. **Assignment Step**: Assign each data point to the nearest centroid:  
   $` C_j = \{x_i : \|x_i - \mu_j\|^2 \leq \|x_i - \mu_l\|^2 \ \forall l\} `$
3. **Update Step**: Recompute the centroid of each cluster based on current assignments.  
4. **Repeat** the assignment and update steps until convergence (i.e., no change in assignments or centroids).

k-Means typically converges to a **local minimum**, and results may vary depending on the initial centroids.

4\. Assumptions
--------------

k-Means assumes:
1\. **Spherical Clusters**: Assumes clusters are isotropic and have similar sizes.  
2\. **Balanced Clusters**: Works best when clusters have roughly equal numbers of points.  
3\. **Continuous Numeric Features**: The algorithm relies on Euclidean distance.  
4\. **Low Noise and Outliers**: Sensitive to noise and extreme values.

5\. Performance Evaluation
-------------------------

Clustering performance can be evaluated using:

- **Inertia (WCSS)**: Measures compactness of clusters. Lower is better.  
- **Silhouette Score**: Measures how similar a point is to its own cluster versus other clusters.  
- **Davies-Bouldin Index**: Ratio of intra-cluster distances to inter-cluster separation.  
- **Elbow Method**: Plots WCSS against $`k`$ to help choose an optimal number of clusters.

6\. Extensions
-------------

- **k-means++**: Improved initialization method to reduce chances of poor convergence.  
- **Mini-Batch k-Means**: Uses small random samples for faster training on large datasets.  
- **Fuzzy c-Means**: Assigns soft cluster memberships instead of hard labels.  
- **Kernel k-Means**: Uses kernel functions to handle non-linearly separable data.

7\. Pros and Cons
----------------

### Pros:
1\. **Simple and Fast**: Easy to implement and computationally efficient.  
2\. **Scalable**: Works well with large datasets.  
3\. **Unsupervised**: No labeled data required.  
4\. **Intuitive Output**: Easy to interpret and visualize.  

### Cons:
1\. **Sensitive to Initialization**: Different initializations can lead to different results.  
2\. **Assumes Equal Cluster Sizes**: May perform poorly with varying cluster densities.  
3\. **Not Robust to Outliers**: Outliers can distort centroid placement.  
4\. **Requires Predefined k**: Must specify the number of clusters beforehand.

