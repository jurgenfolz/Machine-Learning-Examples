Decision Trees
==============

**Decision Trees** are a supervised learning algorithm used for both classification and regression tasks. They are non-parametric models that split the data into subsets based on feature values, resulting in a tree-like structure. Decision trees aim to create a model that predicts the target variable by learning simple decision rules inferred from the data features.

1.  Model Definition

* * * * *

A decision tree consists of the following components:

-   **Root Node**: The top node representing the entire dataset, split into subsets based on the best feature.
-   **Internal Nodes**: Represent a feature-based decision or test.
-   **Leaf Nodes**: Represent the outcome (class label for classification, predicted value for regression).

Each split in a decision tree is based on a feature that best separates the data according to a specific criterion.

### Splitting Criteria

-   **Gini Impurity** (for classification): Measures the impurity of a split. Lower values indicate purer nodes:

    $`Gini = 1 - \sum_{i=1}^{k} p_i^2`$

    where $`p_i`$ is the proportion of instances belonging to class $`i`$.

-   **Entropy (Information Gain)** (for classification): Measures the amount of information gained from a split:

    $`Entropy = - \sum_{i=1}^{k} p_i \log_2(p_i)`$

    Information Gain is calculated as the difference in entropy before and after the split.

-   **Mean Squared Error (MSE)** (for regression): Measures the variance within a node, minimizing the difference between the predicted and actual values.

1.  Cost Function

* * * * *

The goal of a decision tree is to minimize the impurity at each split by choosing the best feature and threshold for splitting. The cost function varies depending on the task:

-   **Classification**: Minimize Gini impurity or maximize information gain.
-   **Regression**: Minimize the sum of squared errors (MSE).

1.  Stopping Criteria

* * * * *

Decision trees grow until all leaf nodes are pure or meet a stopping criterion. Common stopping criteria include:

-   Maximum depth of the tree.
-   Minimum number of samples required to split an internal node.
-   Minimum number of samples required at a leaf node.

1.  Pruning

* * * * *

To prevent overfitting, decision trees are pruned by removing branches that provide little predictive power. Pruning can be done in two ways:

-   **Pre-pruning (Early Stopping)**: Stop growing the tree when it meets specific criteria.
-   **Post-pruning**: Remove branches after the tree is fully grown based on a validation set.

1.  Assumptions

* * * * *

Decision trees make the following assumptions:

1.  **Feature Independence**: Assumes features are independent and contribute equally.

2.  **No Missing Values**: Performs best with complete data.

3.  **Greedy Splitting**: Uses a greedy algorithm to split nodes, aiming for a locally optimal split.

4.  Performance Evaluation

* * * * *

The performance of a decision tree is evaluated using different metrics depending on the task:

-   **Classification**: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
-   **Regression**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).

1.  Extensions

* * * * *

-   **Random Forest**: An ensemble method that builds multiple decision trees and aggregates their predictions.
-   **Gradient Boosting**: Builds trees sequentially, focusing on correcting errors from previous trees.
-   **Extra Trees (Extremely Randomized Trees)**: Introduces randomness in feature selection and splitting to reduce variance.

1.  Pros and Cons

* * * * *

### Pros:

1.  **Simple and Interpretable**: Easy to understand and visualize.
2.  **Non-Parametric**: Makes no assumptions about the underlying data distribution.
3.  **Handles Both Classification and Regression**: Versatile for various tasks.
4.  **Captures Non-Linear Relationships**: Can model complex decision boundaries.
5.  **Robust to Irrelevant Features**: Automatically selects important features.

### Cons:

1.  **Prone to Overfitting**: Can create overly complex trees that do not generalize well.
2.  **Sensitive to Small Variations in Data**: Minor changes can lead to different tree structures.
3.  **Biased with Imbalanced Data**: Needs techniques like class weighting or balancing.
4.  **No Probabilistic Output**: Provides hard class labels without confidence estimates.