k-Nearest Neighbors (k-NN) Algorithm
====================================

The **k-Nearest Neighbors (k-NN)** algorithm is a supervised learning method used for classification and regression tasks. It is a **non-parametric** and **instance-based** learning algorithm, meaning it does not make explicit assumptions about the underlying data distribution and instead stores the training instances for inference.

1\. Classification with k-NN
----------------------------

Given a dataset , where is a feature vector with features, and is the class label among possible classes, the k-NN classification algorithm works as follows:

### 1.1. Distance Computation

To classify a new instance , we compute its distance to all training points. The most common metric used is the **Euclidean distance**, given by:

$`d(x_q, x_i) = \sqrt( sum_{j=1 to d} [ (x_qj - x_ij)^2 ] )`$

where and are the -th components of and , respectively.

Other distance metrics include:

-   **Manhattan Distance**: 

    $`d(x_q, x_i) = sum_{j=1 to d} [ |x_qj - x_ij| ]`$

-   **Minkowski Distance** (generalized form): where gives Euclidean distance and gives Manhattan distance:

    $`d(x_q, x_i) = ( sum_{j=1 to d} [ |x_qj - x_ij|^p ] )^(1/p)`$

-   **Cosine Similarity**: where represents the dot product of the vectors, and , are their respective magnitudes.

    $`d(x_q, x_i) = 1 - ( (x_q · x_i) / ( ||x_q|| * ||x_i|| ) )`$

-   **Chebyshev Distance**: which measures the greatest difference along any coordinate dimension.

    $`d(x_q, x_i) = max_{j in [1..d]} ( |x_qj - x_ij| )`$

### 1.2. Selecting the k Nearest Neighbors

After computing the distances, we select the **k** closest points to , denoted as :

where the indices correspond to the k smallest distances.

### 1.3. Majority Voting

For classification, the predicted class is determined by majority voting:

where is the indicator function:

If there is a tie, a weighted vote based on distances can be used:

and the class with the highest weighted sum is chosen.

2\. Regression with k-NN
------------------------

For regression, the output is computed as the average (or weighted average) of the k nearest neighbors:

or using distance-weighted averaging:

where is typically chosen as the inverse distance weight.

3\. Choosing k
--------------

The choice of is crucial for k-NN's performance:

-   **Small k**: More sensitive to noise, may lead to overfitting.

-   **Large k**: More generalized, may lead to underfitting.

-   A common heuristic is to use , where is the dataset size.

4\. Computational Complexity
----------------------------

-   **Training time complexity**: (since k-NN is instance-based and has no explicit training phase).

-   **Prediction time complexity**: (as we compute distances to all training instances and then sort them).

-   Optimizations such as **KD-Trees** and **Ball Trees** can reduce the complexity for large datasets.

5\. Pros and Cons
----------------------------
#### Pros:

1.  **Simple and Intuitive**: Easy to understand and implement.
2.  **No Training Phase**: k-NN is a lazy learner, meaning it doesn't require a training phase.
3.  **Versatile**: Can be used for both classification and regression tasks.
4.  **Non-parametric**: Makes no assumptions about the underlying data distribution.
5.  **Adaptable**: Can handle multi-class classification problems.

#### Cons:

1.  **Computationally Expensive**: High prediction time complexity, especially with large datasets.
2.  **Memory Intensive**: Requires storing the entire training dataset.
3.  **Sensitive to Irrelevant Features**: Performance can degrade if irrelevant features are present.
4.  **Sensitive to Scale**: Requires feature scaling (normalization or standardization) for good performance.
5.  **Curse of Dimensionality**: Performance can degrade with high-dimensional data.