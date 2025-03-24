Agglomerative Clustering
========================

**Agglomerative Clustering** is a type of **hierarchical clustering** algorithm used in unsupervised learning. It builds nested clusters by **recursively merging** the closest pairs of clusters until all data points are grouped into a single cluster. The result is typically visualized using a **dendrogram**, a tree-like diagram that illustrates the hierarchy of cluster merges.

1\. Model Definition
-------------------

Given a dataset $`X = \{x_1, x_2, ..., x_n\}`$, agglomerative clustering treats each data point as its own singleton cluster at the start. It then follows a **bottom-up** approach:

1. Compute a distance matrix between all clusters.
2. Merge the **two closest clusters**.
3. Recalculate the distance matrix.
4. Repeat steps 2–3 until only one cluster remains.

At each iteration, the algorithm chooses which clusters to merge based on a **linkage criterion**.

### Common Linkage Criteria:
- **Single Linkage**: Distance between the closest points of two clusters.  
  $`D(A, B) = \min_{a \in A, b \in B} \|a - b\|`$
- **Complete Linkage**: Distance between the farthest points.  
  $`D(A, B) = \max_{a \in A, b \in B} \|a - b\|`$
- **Average Linkage**: Average pairwise distance between all points in the clusters.  
  $`D(A, B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} \|a - b\|`$
- **Ward’s Method**: Merges clusters that minimize the increase in total within-cluster variance.

2\. Cost Function
----------------

Agglomerative clustering does not optimize a single explicit cost function like k-means or t-SNE. Instead, it relies on **greedy, local decisions** made by the linkage strategy at each step.

In **Ward’s linkage**, the cost function minimized at each merge is the **sum of squared differences** within all clusters:

$`
\Delta E = \sum_{x \in C} \|x - \mu_C\|^2
`$

where $`\mu_C`$ is the mean of cluster $`C`$.

3\. Algorithm Steps
-------------------

1. Start with each data point as its own cluster.  
2. Compute the distance matrix.  
3. Find the pair of clusters with the smallest distance.  
4. Merge the pair into a new cluster.  
5. Update the distance matrix using the chosen linkage method.  
6. Repeat until all points are merged into a single cluster.

The hierarchy of merges can be visualized using a **dendrogram**, which helps determine a suitable number of clusters by "cutting" the tree at a specific height.

4\. Assumptions
--------------

Agglomerative clustering assumes:
1\. **Meaningful Distance Metric**: Clustering quality depends heavily on the choice of distance metric (e.g., Euclidean, cosine).  
2\. **Hierarchical Structure Exists**: Best suited for data with an inherent hierarchical structure.  
3\. **Complete Data**: Handles missing data poorly without preprocessing.  

5\. Performance Evaluation
-------------------------

Clustering performance can be evaluated using:

- **Dendrogram Analysis**: Visual inspection of merge levels to decide cluster cutoff.  
- **Silhouette Score**: Measures how similar a point is to its own cluster vs. other clusters.  
- **Cophenetic Correlation Coefficient**: Measures how faithfully the dendrogram preserves pairwise distances.  
- **Adjusted Rand Index (ARI)**: Compares predicted clusters to ground truth labels if available.

6\. Extensions
-------------

- **Divisive Clustering**: A top-down hierarchical approach, opposite to agglomerative clustering.  
- **Scikit-learn’s AgglomerativeClustering**: Offers scalability options and memory-efficient implementations.  
- **Hybrid Methods**: Combines hierarchical clustering with other techniques like DBSCAN or k-means.

7\. Pros and Cons
----------------

### Pros:
1\. **No Need to Specify k**: Can determine the number of clusters by analyzing the dendrogram.  
2\. **Flexible Linkage Criteria**: Multiple strategies available for different types of data.  
3\. **Good for Nested Structures**: Captures hierarchical relationships.  
4\. **Deterministic**: Always produces the same results (no randomness).  

### Cons:
1\. **Computationally Expensive**: $`O(n^3)`$ time complexity for naïve implementations.  
2\. **Not Scalable**: Inefficient for very large datasets.  
3\. **Sensitive to Noise and Outliers**: Can distort the clustering structure.  
4\. **Greedy Merges are Irreversible**: Early incorrect merges can't be undone.

Agglomerative clustering is a powerful tool for analyzing hierarchical relationships in data and is especially useful in biology, document clustering, and taxonomy construction.
