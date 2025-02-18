Support Vector Machines (SVM)
=============================

**Support Vector Machines (SVM)** is a supervised learning algorithm used for both classification and regression tasks. It is particularly effective for high-dimensional data and cases where the number of dimensions exceeds the number of samples. SVM aims to find the optimal **hyperplane** that best separates different classes in a dataset.

1\. Model Definition
-------------------

Given a dataset $`D = \{(x_i, y_i)\} `$ where $`x_i `$ is a feature vector and $`y_i`$ in $` {-1, 1} `$ is the class label, SVM finds the **hyperplane** that maximizes the **margin** (distance between the hyperplane and the closest data points, called **support vectors**).

A hyperplane is defined as:

$`
w^T x + b = 0
`$

where:
- \( w \) is the weight vector (normal to the hyperplane)
- \( b \) is the bias term.

The decision rule is:
- \( f(x) > 0 \) classifies the input as \( y = 1 \),
- \( f(x) < 0 \) classifies the input as \( y = -1 \).

### Maximum Margin

The optimal hyperplane is chosen such that it maximizes the **margin** \( M \), given by:

$`
M = \frac{2}{\|w\|}\
`$

This ensures better generalization to unseen data.

2\. Cost Function
----------------

SVM minimizes the following **hinge loss** function:

$`
J(w, b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (w^T x_i + b))
`$

where:
- \( C \) is a regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors.

### Hard Margin vs. Soft Margin

- **Hard Margin SVM**: No misclassification allowed, requires data to be **linearly separable**.
- **Soft Margin SVM**: Allows misclassifications by introducing a slack variable \( \xi \), making SVM robust to noise.

3\. Kernel Trick (Non-Linear SVM)
---------------------------------

For **non-linearly separable data**, SVM can transform the data into a higher-dimensional space using the **kernel trick**, which maps the input features to a new space where a linear separation is possible.

Common kernels include:
- **Linear Kernel**: $` K(x_i, x_j) = x_i^T x_j `$
- **Polynomial Kernel**:  $`K(x_i, x_j) = (x_i^T x_j + c)^d `$
- **Radial Basis Function (RBF) Kernel**:  $`K(x_i, x_j) = e^{-\gamma \|x_i - x_j\|^2} `$
- **Sigmoid Kernel**:  $`K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c) `$

4\. Parameter Estimation\
-----------------------

The key hyperparameters for SVM include:
- **C (Regularization Parameter)**: Controls margin size and misclassification tolerance.
- **Kernel Function**: Determines the feature transformation method.
- **Gamma (gamma)**: Used in RBF and polynomial kernels to control the impact of individual data points.

These parameters are typically tuned using **cross-validation**.

5\. Assumptions
--------------

SVM assumes:
1\. **Data is separable (linearly or using a kernel)**: Works best when the data can be separated by a hyperplane.
2\. **Feature Scaling is Required**: SVM is sensitive to feature magnitudes, so data should be standardized.
3\. **Balanced Data**: Works best when both classes have similar representation.

6\. Performance Evaluation
-------------------------

The performance of an SVM model is evaluated using:
- **Classification**: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- **Regression (SVR)**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).

7\. Extensions
-------------

- **Support Vector Regression (SVR)**: Adapts SVM for regression tasks.
- **One-Class SVM**: Used for outlier detection and anomaly detection.
- **Multi-Class SVM**: Uses strategies like One-vs-One or One-vs-Rest for multi-class problems.

8\. Pros and Cons
----------------

### Pros:
1\. **Effective in High-Dimensional Spaces**: Works well when the number of features is large.
2\. **Robust to Overfitting**: Especially useful in small datasets.
3\. **Works with Non-Linear Data**: Through the use of kernels.
4\. **Maximizes Margin**: Ensures better generalization.

### Cons:
1\. **Computationally Expensive**: Training time increases with large datasets.
2\. **Difficult to Tune**: Kernel and hyperparameter selection can be complex.
3\. **Requires Feature Scaling**: Sensitive to different feature scales.
4\. **Less Interpretable**: Harder to understand compared to decision trees.

Support Vector Machines are widely used in text classification, image recognition, bioinformatics, and financial applications due to their ability to handle high-dimensional and complex data structures.