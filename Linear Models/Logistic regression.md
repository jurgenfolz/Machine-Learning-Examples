Logistic Regression
===================

**Logistic Regression** is a supervised learning algorithm used for binary classification tasks. It models the probability that a given input belongs to a particular class by applying the logistic function (also known as the sigmoid function) to a linear combination of input features.

1.  Model Definition

* * * * *

Given a dataset D={(xi,yi)}D = \{(x_i, y_i)\}D={(xi​,yi​)} where xix_ixi​ is a feature vector and yi∈{0,1}y_i \in \{0, 1\}yi​∈{0,1} is the target variable, logistic regression models the probability that yi=1y_i = 1yi​=1 as:

$`P(y=1∣x)=11+e-(β0+β1x1+β2x2+⋯+βdxd)P(y = 1 | x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_d x_d)}}P(y=1∣x)=1+e-(β0​+β1​x1​+β2​x2​+⋯+βd​xd​)1​´$

where:

-   $`β0\beta_0β0​`$ is the intercept (bias term),
-   $`β1,β2,...,βd\beta_1, \beta_2, ..., \beta_dβ1​,β2​,...,βd`$​ are the regression coefficients (weights),
-   $`x1,x2,...,xdx_1, x_2, ..., x_dx1​,x2​,...,xd​`$ are the feature values.

1.  Cost Function

* * * * *

Unlike linear regression, logistic regression uses the **log-loss (cross-entropy)** as its cost function:

$`J(β)=-1n∑i=1n[yilog⁡(y^i)+(1-yi)log⁡(1-y^i)]J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]J(β)=-n1​∑i=1n​[yi​log(y^​i​)+(1-yi​)log(1-y^​i​)]`$

where:

-   $`y^i\hat{y}_iy^​i​`$ is the predicted probability for the iii-th instance.

1.  Parameter Estimation

* * * * *

The parameters β\betaβ are estimated by minimizing the cost function using an optimization algorithm such as **Gradient Descent**. There is no closed-form solution for logistic regression like in linear regression.

### Gradient Descent Update Rule:

$`βj←βj-α∂J(β)∂βj\beta_j \leftarrow \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}βj​←βj​-α∂βj​∂J(β)`$​ where:

-   $`α\alphaα`$ is the learning rate,
-   $`∂J(β)∂βj\frac{\partial J(\beta)}{\partial \beta_j}∂βj​∂J(β)​`$ is the gradient of the cost function with respect to βj\beta_jβj`$​.

1.  Interpretation of Coefficients

* * * * *

Each coefficient $`βj\beta_jβj`$​ represents the log-odds change in the probability of the target being 1 for a one-unit increase in $`xjx_jxj`$​, assuming other variables are held constant.

The odds ratio is given by: $`eβje^{\beta_j}eβj​`$

1.  Assumptions

* * * * *

Logistic regression makes the following assumptions:

1.  **Linearity of Log-Odds**: The log-odds of the outcome are linearly related to the independent variables.

2.  **Independence of Observations**: The observations are independent of each other.

3.  **No Multicollinearity**: The independent variables should not be highly correlated.

4.  **Large Sample Size**: Logistic regression performs better with larger sample sizes.

5.  Performance Evaluation

* * * * *

The performance of a logistic regression model is typically evaluated using metrics such as:

-   **Accuracy**: $`\frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}​`$
-   **Precision**: $`\frac{\text{True Positives}}{\text{True Positives + False Positives}}​`$
-   **Recall (Sensitivity)**: $`\frac{\text{True Positives}}{\text{True Positives + False Negatives}}​`$
-   **F1 Score**: Harmonic mean of precision and recall.
-   **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Measures the model's ability to distinguish between classes.

1.  Extensions

* * * * *

-   **Multinomial Logistic Regression**: Used for multi-class classification problems.
-   **Regularized Logistic Regression**: Adds regularization terms (L1 for Lasso or L2 for Ridge) to prevent overfitting.

1.  Pros and Cons

* * * * *

### Pros:

1.  **Simple and Interpretable**: Easy to understand and interpret.
2.  **Efficient**: Computationally inexpensive for small to medium datasets.
3.  **Probabilistic Output**: Provides class probabilities instead of just class labels.
4.  **Widely Used**: Commonly used for binary classification tasks.

### Cons:

1.  **Sensitive to Outliers**: Can be influenced by extreme values.
2.  **Linear Decision Boundary**: Cannot capture complex relationships without feature engineering.
3.  **Assumes Independence**: Assumes that the features are independent of each other.
4.  **Requires Balanced Data**: Performs poorly with highly imbalanced datasets.