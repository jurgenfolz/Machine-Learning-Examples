Linear Regression (Ordinary Least Squares)
==========================================

**Linear Regression**, also known as **Ordinary Least Squares (OLS)**, is a fundamental supervised learning algorithm used for regression tasks. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the data.

1.  Model Definition

* * * * *

Given a dataset $`D = \{(x_i, y_i)\}​`$ where $`x_i`$ is a feature vector and $`y_i`$ is the target variable, linear regression assumes a linear relationship:

$`dy= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_d x_d + \epsilon`$

where:

-   $`\beta_0`$ is the intercept (bias term),
-   $`\beta_1, \beta_2, ..., \beta_d`$ are the regression coefficients (weights),
-   $`x_1, x_2, ..., x_d`$ are the feature values,
-   $`\epsilon`$ is the error term, representing noise or unmodeled variation in the data.

1.  Cost Function

* * * * *

The **Ordinary Least Squares (OLS)** method estimates the parameters β\beta by minimizing the sum of squared residuals:

$`J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2`$

where $`y^i\hat{y}_i`$ is the predicted value given by:

$`\hat{y} = X \beta`$

1.  Parameter Estimation

* * * * *

The optimal values for β\beta can be obtained using the **normal equation**:

$`\beta = (X^T X)^{-1} X^T y`$

where:

-   $`X`$ is the design matrix containing the feature values,
-   $`y`$ is the target variable vector.

For large datasets, iterative methods like **Gradient Descent** are preferred over the normal equation due to computational efficiency.

1.  Assumptions

* * * * *

Linear regression relies on the following assumptions:

1.  **Linearity**: The relationship between the independent and dependent variables is linear.

2.  **Independence**: The residuals (errors) are independent of each other.

3.  **Homoscedasticity**: The variance of residuals is constant across all levels of the independent variables.

4.  **Normality**: The residuals follow a normal distribution.

5.  **No multicollinearity**: Independent variables should not be highly correlated with each other.

6.  Performance Evaluation

* * * * *

The performance of a linear regression model is typically evaluated using:

-   **Mean Squared Error (MSE)**:

    $`MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2`$

-   **Root Mean Squared Error (RMSE)**:

    $`RMSE = \sqrt{MSE}`$

-   **R-squared (R2R^2) Score**:

    $`R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}`$

    where $`\bar{y}`$ is the mean of yy.

1.  Extensions

* * * * *

-   **Multiple Linear Regression**: Extends simple linear regression to multiple predictors.
-   **Polynomial Regression**: Models non-linear relationships by introducing polynomial terms.
-   **Ridge Regression (L2 Regularization)**: Adds a penalty term $`\lambda \sum \beta^2`$ to prevent overfitting.
-   **Lasso Regression (L1 Regularization)**: Adds a penalty term $`\lambda \sum |\beta|`$ to enforce sparsity.

1.  Pros and Cons

* * * * *

### Pros:

1.  **Simple and Interpretable**: Provides clear insights into feature relationships.
2.  **Efficient**: Computationally inexpensive for small to medium datasets.
3.  **Widely Used**: Forms the foundation of many advanced statistical and machine learning models.
4.  **Handles Continuous Data Well**: Works well for numerical target variables.

### Cons:

1.  **Sensitive to Outliers**: Can be heavily influenced by extreme values.
2.  **Assumptions Must Hold**: Performance degrades if assumptions like linearity and independence are violated.
3.  **Limited to Linear Relationships**: Cannot capture complex non-linear patterns without modifications.
4.  **Multicollinearity Issues**: Highly correlated independent variables can distort coefficient estimates.

**Ridge Regression** is an extension of linear regression that adds an L2 regularization term to the cost function to prevent overfitting:

where is a regularization parameter that controls the penalty applied to the regression coefficients. Larger values of shrink coefficients towards zero, reducing model complexity.

### Advantages of Ridge Regression:

1.  Helps mitigate multicollinearity.

2.  Reduces model variance and prevents overfitting.

3.  Works well when the number of predictors is large.

### Disadvantages:

1.  Coefficients are shrunk but never reach zero.

2.  Less interpretable compared to standard linear regression.

3.  Lasso Regression (L1 Regularization)

* * * * *

**Lasso Regression** (Least Absolute Shrinkage and Selection Operator) is similar to Ridge Regression but uses an L1 penalty:

This penalty encourages sparsity, meaning some coefficients become exactly zero, effectively performing feature selection.

### Advantages of Lasso Regression:

1.  Performs automatic feature selection by setting some coefficients to zero.

2.  Reduces overfitting by penalizing large coefficients.

3.  Suitable for high-dimensional datasets with many irrelevant features.

### Disadvantages:

1.  Can underperform when features are highly correlated.

2.  Computationally expensive for very large datasets.

3.  Choosing Between Ridge and Lasso

* * * * *

-   **Use Ridge** when all predictors contribute to the outcome and multicollinearity is a concern.

-   **Use Lasso** when feature selection is desired and irrelevant features need to be eliminated.

-   **Elastic Net** is a hybrid approach that combines both Ridge and Lasso penalties.

1.  Pros and Cons

* * * * *

### Pros:

1.  **Simple and Interpretable**: Provides clear insights into feature relationships.

2.  **Efficient**: Computationally inexpensive for small to medium datasets.

3.  **Regularization Helps**: Ridge and Lasso improve generalization and prevent overfitting.

4.  **Feature Selection**: Lasso selects important features automatically.

### Cons:

1.  **Sensitive to Outliers**: Still influenced by extreme values.

2.  **Requires Tuning**: The regularization parameter must be carefully selected.

3.  **Interpretability Issues**: Ridge does not set coefficients to zero, making it less interpretable than Lasso.