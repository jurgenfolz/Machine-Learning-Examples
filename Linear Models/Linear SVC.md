Linear Support Vector Classifier (Linear SVC)
=============================================

**Linear Support Vector Classifier (Linear SVC)** is a supervised learning algorithm used for classification tasks. It aims to find the hyperplane that best separates the data into distinct classes by maximizing the margin between the classes. Unlike Logistic Regression, Linear SVC focuses on finding the maximum-margin hyperplane, making it robust to noise and outliers.

1.  Model Definition

* * * * *

Given a dataset $`D={(xi,yi)}D = \{(x_i, y_i)\}D={(xi​,yi​)}​`$ where $`xix_ixi​​`$ is a feature vector and $`yi∈{-1,1}y_i \in \{-1, 1\}yi​∈{-1,1}​`$ is the class label, the goal of Linear SVC is to find a linear decision boundary (hyperplane) defined by:

$´f(x)=wTx+b=0f(x) = w^T x + b = 0f(x)=wTx+b=0​`$

where:

-   $`w`$ is the weight vector (coefficients),
-   $`b​`$ is the bias (intercept term).

The decision rule is:

-   $`f(x)>0f(x) > 0f(x)>0​`$ classifies the input as $`y=1y = 1y=1​`$,
-   $`f(x)<0f(x) < 0f(x)<0​`$ classifies the input as $`y=-1y = -1y=-1​`$.

1.  Cost Function

* * * * *

Linear SVC uses the **Hinge Loss** as its cost function, which penalizes misclassified points and points within the margin:

$`J(w,b)=12∥w∥2+C∑i=1nmax⁡(0,1-yi(wTxi+b))J(w, b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (w^T x_i + b))J(w,b)=21​∥w∥2+C∑i=1n​max(0,1-yi​(wTxi​+b))

where:

-   $`CCC is the regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors.
-   The term $`12∥w∥2\frac{1}{2} \|w\|^221​∥w∥2 is the regularization component that ensures a large margin.

1.  Parameter Estimation

* * * * *

The optimal values for www and bbb are obtained by solving a constrained optimization problem:

$`min⁡w,b12∥w∥2+C∑i=1nmax⁡(0,1-yi(wTxi+b))\min_{w, b} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (w^T x_i + b))w,bmin​21​∥w∥2+Ci=1∑n​max(0,1-yi​(wTxi​+b))

This optimization problem is typically solved using techniques like **Quadratic Programming** or **Stochastic Gradient Descent**.

1.  Interpretation of Coefficients

* * * * *

Each coefficient in the weight vector www represents the importance of the corresponding feature in determining the classification boundary. Larger values indicate stronger influence.

1.  Assumptions

* * * * *

Linear SVC makes the following assumptions:

1.  **Linearly Separable Data**: Performs best when the data is approximately linearly separable.

2.  **Independence of Observations**: Assumes that the data points are independent of each other.

3.  **No Multicollinearity**: Features should not be highly correlated.

4.  **Feature Scaling**: Requires feature scaling (e.g., normalization or standardization) for optimal performance.

5.  Performance Evaluation

* * * * *

The performance of a Linear SVC model is typically evaluated using:

-   **Accuracy**: $`\frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}Total Number of PredictionsNumber of Correct Predictions​
-   **Precision**: $`\frac{\text{True Positives}}{\text{True Positives + False Positives}}True Positives + False PositivesTrue Positives​
-   **Recall (Sensitivity)**: $`\frac{\text{True Positives}}{\text{True Positives + False Negatives}}True Positives + False NegativesTrue Positives​
-   **F1 Score**: Harmonic mean of precision and recall.
-   **ROC-AUC**: Measures the model's ability to distinguish between classes.

1.  Extensions

* * * * *

-   **Kernel SVM**: Extends Linear SVC by using kernel functions (e.g., polynomial or RBF kernels) to capture non-linear relationships.
-   **Multiclass SVM**: Linear SVC can be extended to handle multi-class problems using strategies like One-vs-One or One-vs-Rest.

1.  Pros and Cons

* * * * *

### Pros:

1.  **Maximizes the Margin**: Results in a more robust and generalizable model.
2.  **Effective in High-Dimensional Spaces**: Performs well with a large number of features.
3.  **Works with Sparse Data**: Suitable for text classification and other sparse datasets.
4.  **Regularization**: The CCC parameter controls the complexity of the model.

### Cons:

1.  **Sensitive to Feature Scaling**: Requires preprocessing of data (normalization or standardization).
2.  **Not Suitable for Non-Linear Data**: Fails to capture non-linear patterns without a kernel.
3.  **Computational Complexity**: Can be slow for very large datasets.
4.  **No Probabilistic Interpretation**: Unlike Logistic Regression, it does not provide class probabilities.