DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
====================================================================

**DBSCAN** is an unsupervised clustering algorithm that groups together points that are closely packed (i.e., high density regions) while marking points in low-density regions as **outliers**. Unlike k-means and hierarchical clustering, DBSCAN does not require specifying the number of clusters and can find arbitrarily shaped clusters.

1\. Model Definition
-------------------

Given a dataset $`X = \{x_1, x_2, ..., x_n\}`$, DBSCAN groups points into clusters based on two parameters:

- $`\varepsilon`$ (epsilon): The maximum distance between two samples for them to be considered as part of the same neighborhood.
- $`minPts`$: The minimum number of points required to form a dense region.

DBSCAN categorizes each point as one of the following:

- **Core Point**: Has at least `minPts` neighbors within $`\varepsilon`$.
- **Border Point**: Has fewer than `minPts` neighbors but is in the neighborhood of a core point.
- **Noise Point** (Outlier): Not a core or border point.

### Definitions:

- **Neighborhood** of a point $`x_i`$:  
  $`N(x_i) = \{x_j \in X \ | \ \|x_i - x_j\| \leq \varepsilon \}`$
  
- **Directly Density-Reachable**: $`x_j`$ is directly density-reachable from $`x_i`$ if $`x_j \in N(x_i)`$ and $`x_i`$ is a core point.

- **Density-Reachable**: A point is density-reachable from another if there exists a chain of directly density-reachable points between them.

2\. Cost Function
----------------

DBSCAN does not explicitly minimize a cost function. Instead, it clusters data by identifying dense regions based on the neighborhood density defined by $`\varepsilon`$ and `minPts`.

The algorithm focuses on **local density** rather than optimizing a global objective.

3\. Algorithm Steps
-------------------

1. For each unvisited point $`x_i`$ in the dataset:
   - Mark $`x_i`$ as visited.
   - Retrieve its $`\varepsilon`$-neighborhood.
   - If the neighborhood has at least `minPts` points, start a new cluster.
   - Expand the cluster by recursively visiting all density-reachable points.
2. If a point is not density-reachable from any other point, label it as noise.
3. Repeat until all points are visited.

4\. Assumptions
--------------

DBSCAN assumes:
1\. **Clusters are Dense**: Points in the same cluster are close together and separated by regions of lower density.  
2\. **Distance Metric is Meaningful**: Typically Euclidean, but other metrics can be used.  
3\. **Consistent Density**: Works best when clusters have similar densities.  

5\. Performance Evaluation
-------------------------

DBSCAN clustering performance can be evaluated using:

- **Silhouette Score**: Measures how well a point fits within its cluster.  
- **Adjusted Rand Index (ARI)**: Measures agreement with ground truth if available.  
- **Cluster Purity**: Evaluates the degree to which clusters contain a single class.  
- **Number of Detected Clusters**: The algorithm dynamically identifies the number of clusters.

6\. Extensions
-------------

- **HDBSCAN**: A hierarchical version of DBSCAN that handles varying densities.  
- **OPTICS**: Orders points to create a reachability plot, avoiding the need to select $`\varepsilon`$ manually.  
- **DBSCAN++**: Improves runtime for large-scale datasets.

7\. Pros and Cons
----------------

### Pros:
1\. **No Need to Specify k**: Automatically detects the number of clusters.  
2\. **Robust to Outliers**: Naturally identifies and excludes noise points.  
3\. **Can Find Arbitrarily Shaped Clusters**: Not limited to spherical clusters.  
4\. **Simple Intuition**: Based on density and local neighborhoods.  

### Cons:
1\. **Parameter Sensitivity**: Choosing appropriate $`\varepsilon`$ and `minPts` is crucial.  
2\. **Fails with Varying Densities**: Struggles when cluster densities differ significantly.  
3\. **Not Scalable to Very Large Datasets**: Naive implementations have high time complexity.  
4\. **Distance Metric Dependency**: Performance depends on the choice of distance function and feature scaling.

